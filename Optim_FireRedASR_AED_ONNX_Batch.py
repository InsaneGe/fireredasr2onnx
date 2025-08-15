import os
import torch
import onnx
from onnxslim import slim
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

encoder_path = os.path.join(args.input, "FireRedASR_AED_L-Encoder-Batch.onnx")
decoder_path = os.path.join(args.input, "FireRedASR_AED_L-Decoder-Batch.onnx")

encoder_slim_path = os.path.join(args.output, "FireRedASR_AED_L-Encoder-Batch.onnx")
decoder_slim_path = os.path.join(args.output, "FireRedASR_AED_L-Decoder-Batch.onnx")

encoder = onnx.load(encoder_path)
decoder = onnx.load(decoder_path)


slim(
    model=encoder,
    output_model=encoder_slim_path,
    no_shape_infer=False,
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)

slim(
    model=decoder,
    output_model=decoder_slim_path,
    no_shape_infer=False,
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)
