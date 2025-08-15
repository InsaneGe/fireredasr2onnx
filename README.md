# FireRed2ONNX

## Introduction

convert FireRedASR-AED to ONNX format with batch inference. accelerate inference and maintain the original ASR performance.

## Getting started

1. Create and Activate the conda environment

```bash
conda create -n asr_export python=3.12
conda activate asr_export

```

2. Install dependencies

```bash
pip install -r requirements.txt
```
note: onnxruntime-gpu 1.22.0 need glibc >= 2.27

3. Download or Prepare FireRedASR-AED weights, e.g.,

```bash
huggingface-cli download FireRedTeam/FireRedASR-AED-L --local-dir ./weights/FireRedASR-AED-L
```

4. Export FireRedASR-ASR to ONNX (Save to `onnx_folder_path`)
```bash
python Export_FireRedASR_AED_Batch.py --model_path ./weights/FireRedASR-AED-L --project_path ./FireRedASR --onnx_folder_path ./onnx_model
```

5. (Optional, with limited improvement) Optim exported ONNX models by `ONNXSlim`(Save to `./onnx_model` by default)
```bash
python Optim_FireRedASR_AED_ONNX_Batch.py --input onnx_model --output onnx_slim
```

6. Inference with CUDA

```bash
python Inference_FireRedASR_AED_ONNX_Batch.py --model_path ./weights/FireRedASR-AED-L --project_path ./FireRedASR --onnx_folder_path ./onnx_slim --batch_size 4
```

## Reference

- https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX/tree/main/FireRedASR
- https://github.com/FireRedTeam/FireRedASR
