import os
import gc
import sys
import shutil
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from pydub import AudioSegment

import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='FireRedASR ONNX Exporter and Runner')
parser.add_argument('--model_path', type=str, default='/home/ec2-user/SageMaker/FireRedTeam/FireRedASR-AED-L', help='Path to the model weights')
parser.add_argument('--project_path', type=str, default="./FireRedASR", help='Path to the FireRedASR project')
parser.add_argument('--onnx_folder_path', type=str, default="./onnx_model", help='Path to save ONNX models')
parser.add_argument('--test_audio', nargs='+', default=["./example/zh_1.wav", "./example/zh_2.wav"], help='List of test audio files')

args = parser.parse_args()

project_path = args.project_path
model_path = args.model_path
onnx_folder_path = args.onnx_folder_path
test_audio = args.test_audio

if project_path not in sys.path:
    sys.path.append(project_path)

from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.data.asr_feat import ASRFeatExtractor

os.makedirs(onnx_folder_path, exist_ok=True)
onnx_model_A = os.path.join(onnx_folder_path, "FireRedASR_AED_L-Encoder-Batch.onnx")
onnx_model_B = os.path.join(onnx_folder_path, "FireRedASR_AED_L-Decoder-Batch.onnx")


DYNAMIC_AXES = True
DUMMY_BATCH_SIZE = 4
DUMMY_INPUT_AUDIO_LENGTH = 200
SAMPLE_RATE = 16000
MAX_SEQ_LEN = 64
STOP_TOKEN = [4]

class FIRE_RED_ENCODER(torch.nn.Module):
    def __init__(self, fire_red):
        super(FIRE_RED_ENCODER, self).__init__()
        self.model = fire_red
        self.model.encoder.positional_encoding.pe.data = self.model.encoder.positional_encoding.pe.data.half()
        self.save_en_keys = [None] * self.model.decoder.n_layers
        self.save_en_values = [None] * self.model.decoder.n_layers

    def forward(self, audio, mask):
        batch_size = audio.shape[0]
        enc_outputs = self.model.encoder(audio, mask)
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            self.save_en_keys[idx] = decoder_layer.cross_attn.w_ks(enc_outputs).view(batch_size, -1, decoder_layer.cross_attn.n_head, decoder_layer.cross_attn.d_k).permute(0, 2, 3, 1)
            self.save_en_values[idx] = decoder_layer.cross_attn.w_vs(enc_outputs).view(batch_size, -1, decoder_layer.cross_attn.n_head, decoder_layer.cross_attn.d_k).permute(0, 2, 1, 3)
        return *self.save_en_keys, *self.save_en_values


class FIRE_RED_DECODER(torch.nn.Module):
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_DECODER, self).__init__()
        self.model = fire_red
        self.num_layers_de_2 = self.model.decoder.n_layers + self.model.decoder.n_layers
        self.num_layers_de_2_plus = self.num_layers_de_2 + 3
        self.num_layers_de_3_plus = self.num_layers_de_2_plus + self.model.decoder.n_layers
        self.max_seq_len = max_seq_len
        self.model.decoder.tgt_word_emb.weight.data *= self.model.decoder.scale
        self.model.decoder.positional_encoding.pe.data = self.model.decoder.positional_encoding.pe.data[:, :max_seq_len].half()
        self.save_de_keys = [None] * self.model.decoder.n_layers
        self.save_de_values = [None] * self.model.decoder.n_layers

    def forward(self, *all_inputs):
        batch_size = all_inputs[0].shape[0]
        input_ids = all_inputs[self.num_layers_de_2]
        history_len = all_inputs[self.num_layers_de_2 + 1]
        ids_len = all_inputs[self.num_layers_de_2 + 2]
        kv_seq_len = history_len + ids_len

        # 使用广播来处理批处理
        pos_indices = torch.arange(ids_len.max(), device=input_ids.device).expand(batch_size)
        pe_indices = history_len + pos_indices

        # 检查 pe 的形状并适应批处理
        pe = self.model.decoder.positional_encoding.pe[:, pe_indices, :]
        pe = pe.transpose(0, 1).float()
        hidden_state = self.model.decoder.tgt_word_emb(input_ids) + pe

        max_kv_seq_len = kv_seq_len.max()
        max_ids_len = ids_len.max()

        # 创建基础的attention mask
        attention_mask = (1 - torch.tril(torch.ones([batch_size, 1, self.max_seq_len, self.max_seq_len], dtype=torch.float32))) * (-2**30)
        attention_mask = attention_mask[:, :, :max_ids_len, :max_kv_seq_len]

        ids_mask = torch.arange(max_ids_len, device=ids_len.device)[None, None, :, None] < ids_len[:, None, None, None]
        kv_mask = torch.arange(max_kv_seq_len, device=kv_seq_len.device)[None, None, None, :] < kv_seq_len[:, None, None, None]
        padding_mask = ids_mask & kv_mask
        
        all_inputs_mask = all_inputs[-1][:, None, None, None].expand(-1, 1, max_ids_len, max_kv_seq_len)
        attention_mask = torch.where(padding_mask, attention_mask * all_inputs_mask, attention_mask)

        attention_mask = attention_mask.float()


        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            hidden_state_norm = decoder_layer.self_attn_norm(hidden_state)
            q = decoder_layer.self_attn.w_qs(hidden_state_norm).view(batch_size, -1, decoder_layer.self_attn.n_head, decoder_layer.self_attn.d_k).transpose(1, 2)
            k = decoder_layer.self_attn.w_ks(hidden_state_norm).view(batch_size, -1, decoder_layer.self_attn.n_head, decoder_layer.self_attn.d_k).permute(0, 2, 3, 1)
            v = decoder_layer.self_attn.w_vs(hidden_state_norm).view(batch_size, -1, decoder_layer.self_attn.n_head, decoder_layer.self_attn.d_k).transpose(1, 2)
            k = torch.cat((all_inputs[idx], k), dim=3)
            v = torch.cat((all_inputs[idx + self.model.decoder.n_layers], v), dim=2)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            hidden_state_attn = decoder_layer.self_attn.fc(torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v).transpose(1, 2).contiguous().view(batch_size, -1, decoder_layer.self_attn.d_model))
            hidden_state_attn += hidden_state
            q = decoder_layer.cross_attn.w_qs(decoder_layer.cross_attn_norm(hidden_state_attn)).view(batch_size, -1, decoder_layer.cross_attn.n_head, decoder_layer.cross_attn.d_k).transpose(1, 2)
            hidden_state_cross = decoder_layer.cross_attn.fc(torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.num_layers_de_2_plus]), dim=-1), all_inputs[idx + self.num_layers_de_3_plus]).transpose(1, 2).contiguous().view(batch_size, -1, decoder_layer.cross_attn.d_model))
            hidden_state_cross += hidden_state_attn
            hidden_state = hidden_state_cross + decoder_layer.mlp(decoder_layer.mlp_norm(hidden_state_cross))
        max_logit_idx = torch.argmax(self.model.decoder.tgt_word_prj(self.model.decoder.layer_norm_out(hidden_state)[:, -1]), dim=-1, keepdim=True).int()
        return *self.save_de_keys, *self.save_de_values, max_logit_idx, kv_seq_len


print('\nStart to export the Encoder part.\n')
with torch.inference_mode():
    if 'aed' in model_path or 'AED' in model_path or 'Aed' in model_path:
        model = FireRedAsr.from_pretrained("aed", model_path)
        model = model.model.float()
        HIDDEN_SIZE = model.encoder.odim
        NUM_HEAD_EN = model.encoder.layer_stack._modules['0'].mhsa.n_head
        NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].self_attn.n_head
        NUM_LAYER_DE = model.decoder.n_layers
        HEAD_DIM_EN = model.encoder.layer_stack._modules['0'].mhsa.d_k
        HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].self_attn.d_k

        scaling = float(HEAD_DIM_DE ** -0.25)
        for i in model.decoder.layer_stack._modules:
            model.decoder.layer_stack._modules[i].self_attn.w_qs.weight.data *= scaling
            model.decoder.layer_stack._modules[i].self_attn.w_qs.bias.data *= scaling
            model.decoder.layer_stack._modules[i].self_attn.w_ks.weight.data *= scaling

        scaling = float(model.decoder.layer_stack._modules['0'].cross_attn.d_k ** -0.25)
        for i in model.decoder.layer_stack._modules:
            model.decoder.layer_stack._modules[i].cross_attn.w_qs.weight.data *= scaling
            model.decoder.layer_stack._modules[i].cross_attn.w_qs.bias.data *= scaling
            model.decoder.layer_stack._modules[i].cross_attn.w_ks.weight.data *= scaling


        fire_red_encoder = FIRE_RED_ENCODER(model)

        output_names = []


        audio = torch.ones((DUMMY_BATCH_SIZE, DUMMY_INPUT_AUDIO_LENGTH, 80), dtype=torch.float32)
        mask = torch.ones((DUMMY_BATCH_SIZE, 1, DUMMY_INPUT_AUDIO_LENGTH), dtype=torch.uint8)

        dynamic_axes = {'audio': {0: 'batch_size', 1: 'audio_len'}, 'mask': {0: 'batch_size', 2: 'audio_len'}, }
        for i in range(NUM_LAYER_DE):
            name = f'en_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 3: 'signal_len'}
        for i in range(NUM_LAYER_DE):
            name = f'en_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 2: 'signal_len'}

        torch.onnx.export(
            fire_red_encoder,
            (audio, mask, ),
            onnx_model_A,
            input_names=['audio', 'mask'],
            output_names=output_names,
            dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
            do_constant_folding=True,
            opset_version=17,
        )
        del fire_red_encoder
        del audio
        del name
        del output_names
        del dynamic_axes
        gc.collect()
        print("\nExport Done!\n\nStart to export the Decoder part.")

        fire_red_decoder = FIRE_RED_DECODER(model, MAX_SEQ_LEN)
        input_ids = torch.tensor([[3]] * DUMMY_BATCH_SIZE, dtype=torch.int32)
        ids_len = torch.tensor([input_ids.shape[-1]] * DUMMY_BATCH_SIZE, dtype=torch.int64)
        history_len = torch.tensor([0] * DUMMY_BATCH_SIZE, dtype=torch.int64)
        save_encoder_key = torch.zeros((DUMMY_BATCH_SIZE, NUM_HEAD_EN, HEAD_DIM_EN, DUMMY_INPUT_AUDIO_LENGTH), dtype=torch.float32)
        save_encoder_value = torch.zeros((DUMMY_BATCH_SIZE, NUM_HEAD_EN, DUMMY_INPUT_AUDIO_LENGTH, HEAD_DIM_EN), dtype=torch.float32)
        past_key_de = torch.zeros((DUMMY_BATCH_SIZE, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
        past_value_de = torch.zeros((DUMMY_BATCH_SIZE, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
        attention_mask = torch.tensor([1] * DUMMY_BATCH_SIZE, dtype=torch.int8)

        input_names = []
        all_inputs = []
        output_names = []
        dynamic_axes = {}

        for i in range(NUM_LAYER_DE):
            name = f'in_de_key_{i}'
            input_names.append(name)
            all_inputs.append(past_key_de)
            dynamic_axes[name] = {0: 'batch_size', 3: 'history_len'}
            name = f'out_de_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 3: 'history_len_plus_ids_len'}
        for i in range(NUM_LAYER_DE):
            name = f'in_de_value_{i}'
            input_names.append(name)
            all_inputs.append(past_value_de)
            dynamic_axes[name] = {0: 'batch_size', 2: 'history_len'}
            name = f'out_de_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 2: 'history_len_plus_ids_len'}

        input_names.append('input_ids')
        all_inputs.append(input_ids)
        dynamic_axes['input_ids'] = {0: 'batch_size'}
        
        input_names.append('history_len')
        all_inputs.append(history_len)
        dynamic_axes['history_len'] = {0: 'batch_size'}
        
        input_names.append('ids_len')
        all_inputs.append(ids_len)
        dynamic_axes['ids_len'] = {0: 'batch_size'}

        for i in range(NUM_LAYER_DE):
            name = f'en_key_{i}'
            input_names.append(name)
            all_inputs.append(save_encoder_key)
            dynamic_axes[name] = {0: 'batch_size', 3: 'signal_len'}
        for i in range(NUM_LAYER_DE):
            name = f'en_value_{i}'
            input_names.append(name)
            all_inputs.append(save_encoder_value)
            dynamic_axes[name] = {0: 'batch_size', 2: 'signal_len'}

        input_names.append('attention_mask')
        all_inputs.append(attention_mask)
        dynamic_axes['attention_mask'] = {0: 'batch_size'}
        
        output_names.append('max_logit_id')
        dynamic_axes['max_logit_id'] = {0: 'batch_size'}
        
        output_names.append('kv_seq_len')
        dynamic_axes['kv_seq_len'] = {0: 'batch_size'}

        torch.onnx.export(
            fire_red_decoder,
            tuple(all_inputs),
            onnx_model_B,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
            do_constant_folding=True,
            opset_version=17,
        )
        del model
        del fire_red_decoder
        del input_ids
        del ids_len
        del history_len
        del save_encoder_key
        del save_encoder_value
        del past_key_de
        del past_value_de
        del attention_mask
        del input_names
        del output_names
        del dynamic_axes
    else:
        print("Currently, only support the FireRedASR-AED")

if project_path in sys.path:
    sys.path.remove(project_path)

print('\nExport done!\n\nStart to run FireRedASR by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
output_names_A = []
for i in range(len(out_name_A)):
    output_names_A.append(out_name_A[i].name)

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
input_names_B = []
output_names_B = []
amount_of_outputs = len(out_name_B)
for i in range(len(in_name_B)):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs):
    output_names_B.append(out_name_B[i].name)

generate_limit = MAX_SEQ_LEN - 1  # 1 = length of input_ids
num_layers = (amount_of_outputs - 2) // 2
num_layers_2 = num_layers + num_layers
num_layers_4 = num_layers_2 + num_layers_2
num_layers_2_plus_1 = num_layers_2 + 1
num_layers_2_plus_2 = num_layers_2 + 2

tokenizer = ChineseCharEnglishSpmTokenizer(model_path + "/dict.txt", model_path + "/train_bpe1000.model")
feat_extractor = ASRFeatExtractor(os.path.join(model_path, "cmvn.ark"))

# 批量处理音频
batch_size = 2  # 设置批量大小
for i in range(0, len(test_audio), batch_size):
    batch_audio = test_audio[i:i+batch_size]
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nProcessing batch: {batch_audio}")

    # 开始运行FireRedASR
    input_ids = np.array([[3]] * len(batch_audio), dtype=np.int32)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]] * len(batch_audio), dtype=np.int64), 'cpu', 0)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, 'cpu', 0)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0] * len(batch_audio), dtype=np.int64), 'cpu', 0)
    attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1] * len(batch_audio), dtype=np.int8), 'cpu', 0)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((len(batch_audio), ort_session_B._inputs_meta[0].shape[1], ort_session_B._inputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', 0)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((len(batch_audio), ort_session_B._inputs_meta[num_layers].shape[1], 0, ort_session_B._inputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', 0)
    layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 3

    input_feed_B = {
        in_name_B[-1].name: attention_mask,
        in_name_B[num_layers_2].name: input_ids,
        in_name_B[num_layers_2_plus_1].name: history_len,
        in_name_B[num_layers_2_plus_2].name: ids_len
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i].name] = past_keys_B
    for i in range(num_layers, num_layers_2):
        input_feed_B[in_name_B[i].name] = past_values_B
    num_decode = 0
    save_token = [[] for _ in range(len(batch_audio))]
    start_time = time.time()

    # 加载和预处理音频
    audios, input_lengths, _ = feat_extractor(batch_audio)
    batch_size = len(batch_audio)
    pad_zeros = torch.zeros(batch_size, 6, 80, dtype=torch.float32, device=audios.device)
    padded_input = torch.cat((audios, pad_zeros), dim=1)
    N, T = padded_input.size()[:2]
    padded_input = np.array(padded_input)
    input_lengths = np.array(input_lengths)
    mask = np.ones((N, 1, T))
    for i in range(N):
        mask[i, 0, input_lengths[i]:] = 0
    mask = mask.astype(np.uint8)

    all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(padded_input, 'cpu', 0),
                                                                       in_name_A1: onnxruntime.OrtValue.ortvalue_from_numpy(mask, 'cpu', 0)})
    for i in range(num_layers_2):
        input_feed_B[in_name_B[layer_indices[i]].name] = all_outputs_A[i]
    
    while num_decode < generate_limit:
        all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-2])
        num_decode += 1
        if all(id in STOP_TOKEN for id in max_logit_ids):
            break
        for i in range(amount_of_outputs):
            input_feed_B[in_name_B[i].name] = all_outputs_B[i]
        if num_decode < 2:
            input_feed_B[in_name_B[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0] * len(batch_audio), dtype=np.int8), 'cpu', 0)
        for j, id in enumerate(max_logit_ids):
            save_token[j].append(id)

    count_time = time.time() - start_time

    for j, tokens in enumerate(save_token):
        text = ""
        for id in tokens:
            token = tokenizer.dict[int(id[0])]
            if int(id[0]) in STOP_TOKEN:
                break
            text += token
        text = text.replace(tokenizer.SPM_SPACE, ' ').strip()
        print(f"\nASR Result for audio {batch_audio[j]}:\n{text}")

print("\nAll processing completed.")
