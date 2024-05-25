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
    - Clear old batches on generation
    - Fixed bucketing and merging
    - Fix tokens per second for batching and duplicates
    - Optimize trainer
    - Consider splitting input and target files for finetune
    - Large batches not performing as well as small (eos?)
    - validation etc batches
    - Get a better more diverse dataset
    - Implement LORA (model freezing, specifying layers)
    - Implement QLora (quantized)
    - Implement CLR
    - Infer from saved adapters
    - load checkpoint and infer script
1 - Modify infer to be interactive
2 - Clean up the inference utilities
2 - Modify model to support bloat16
1 - Generate large file generator for jsonl
1 - Memory Calculator

Calvin Run (no lora, 3b code)
402/403 [37:56<00:05,  5.66s/batch, Batch Loss=0.774, Batch Tokens=64, Batch Time=5.814s]

