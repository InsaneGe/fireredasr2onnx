import os
import sys
import time
import numpy as np
import torch
import onnxruntime
from pydub import AudioSegment

import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='Path to the model weights')
parser.add_argument('--project_path', type=str, default="./FireRedASR", help='Path to the FireRedASR project')
parser.add_argument('--onnx_path', type=str, default="./onnx_model", help='Path to save ONNX models')
parser.add_argument('--test_audio', nargs='+', default=['./example/test1.wav'] * 100, help='List of test audio files')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
args = parser.parse_args()

if args.project_path not in sys.path:
    sys.path.append(args.project_path)
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.data.asr_feat import ASRFeatExtractor

onnx_model_A = os.path.join(args.onnx_path, "FireRedASR_AED_L-Encoder-Batch.onnx")
onnx_model_B = os.path.join(args.onnx_path, "FireRedASR_AED_L-Decoder-Batch.onnx")

# If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
ORT_Accelerate_Providers = ['CUDAExecutionProvider']
MAX_THREADS = 4
DEVICE_ID = 0
MAX_SEQ_LEN = 64
SAMPLE_RATE = 16000
STOP_TOKEN = [4]
BATCH_SIZE = args.batch_size  # 批处理大小设置


if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    device_type = 'cuda'
else:
    device_type = 'cpu'


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True
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

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
output_names_A = []
for i in range(len(out_name_A)):
    output_names_A.append(out_name_A[i].name)

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers)
model_dtype = ort_session_B._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
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

tokenizer = ChineseCharEnglishSpmTokenizer(args.model_path + "/dict.txt", args.model_path + "/train_bpe1000.model")
feat_extractor = ASRFeatExtractor(os.path.join(args.model_path, "cmvn.ark"))

test_audio = args.test_audio
for i in range(0, len(test_audio), BATCH_SIZE):
    batch_audio = test_audio[i:min(len(test_audio),i+BATCH_SIZE)] # in case len(test_audio) can't be divided by batch_size
    start_time = time.time()

    input_ids = np.array([[3]] * len(batch_audio), dtype=np.int32)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]] * len(batch_audio), dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0] * len(batch_audio), dtype=np.int64), device_type, DEVICE_ID)
    attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1] * len(batch_audio), dtype=np.int8), device_type, DEVICE_ID)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((len(batch_audio), ort_session_B._inputs_meta[0].shape[1], ort_session_B._inputs_meta[0].shape[2], 0), dtype=model_dtype), device_type, DEVICE_ID)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((len(batch_audio), ort_session_B._inputs_meta[num_layers].shape[1], 0, ort_session_B._inputs_meta[num_layers].shape[3]), dtype=model_dtype), device_type, DEVICE_ID)
    layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 3
    input_feed_B = {
        input_names_B[-1]: attention_mask,
        input_names_B[num_layers_2]: input_ids,
        input_names_B[num_layers_2_plus_1]: history_len,
        input_names_B[num_layers_2_plus_2]: ids_len
    }
    for j in range(num_layers):
        input_feed_B[input_names_B[j]] = past_keys_B
    for j in range(num_layers, num_layers_2):
        input_feed_B[input_names_B[j]] = past_values_B
    num_decode = 0
    save_token = [[] for _ in range(len(batch_audio))]

    # 加载和预处理音频
    before_process = time.time()
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

    all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(padded_input, device_type, DEVICE_ID),
                                                                       in_name_A1: onnxruntime.OrtValue.ortvalue_from_numpy(mask, device_type, DEVICE_ID)})
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
            input_feed_B[in_name_B[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0] * len(batch_audio), dtype=np.int8), device_type, DEVICE_ID)
        for j, id in enumerate(max_logit_ids):
            save_token[j].append(id)
    for j, tokens in enumerate(save_token):
        text = ""
        for id in tokens:
            token = tokenizer.dict[int(id[0])]
            if int(id[0]) in STOP_TOKEN:
                break
            text += token
        text = text.replace(tokenizer.SPM_SPACE, ' ').strip()
        audio_path = batch_audio[j]
        last_dir = audio_path.split(os.sep)[-2]
        print(f"{last_dir}/{os.path.basename(audio_path)}: {text}")

if args.project_path in sys.path:
    sys.path.remove(args.project_path)