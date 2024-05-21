# Introduction
This is a ground up rework of an MLX training script.

## Intallation
To get started, install using pip

```bash
pip install -r requirements.txt
```

## Inference
The folloing scripts show how to prompt a model

```python
python infer.py --model ibm-granite/granite-3b-code-instruct --prompt "write a fibonacci function in python"
```

## Supported Models
The following model familes are supported.

- Meta Llama-3
- Meta Llama-2
- Mistral-7B
- IBM Granite
- IBM Granite Code Models

TODO
--------
1 - Pretrainer
    - implement batch sorter thingy
    - Clear old batches on generation
    - Implement Warm Up
    - Implement LORA (model freezing, specifying layers)
    - Implement QLora (quantized)
    - Save Adapters
    - Infer from saved adapters
    - load checkpoint and infer script
    - Fix Padding Issue
1 - Modify infer to be interactive
2 - Clean up the inference utilities
2 - Modify model to support bloat16
1 - Generate large file generator for jsonl

Calvin Run (no lora, 3b code)
402/403 [37:56<00:05,  5.66s/batch, Batch Loss=0.774, Batch Tokens=64, Batch Time=5.814s]

