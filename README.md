# Introduction
This is a ground up rework of an MLX training script.

## Intallation
To get started, install using pip

```bash
pip install -r requirements.txt
```

## Inference
The following scripts show how to prompt a model

```python
python infer.py --model ibm-granite/granite-3b-code-instruct --prompt "write a fibonacci function in python"
```

TODO
--------
1 - Update Lazyfox to use BOS and EOS to improve batching
2 - Fix Fine Tuning
1 - Models
    - Test new version of mistral
2 - Perform a light fine tune, with a inference load using Mistral
3 - Training
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
2 - Modify model to support bloat16
1 - Generate large file generator for jsonl
1 - Memory Calculator
4 - Refactor and Fix Lazyfox
5 - Refactor and Fix Pretrain

Calvin Run (no lora, 3b code)
402/403 [37:56<00:05,  5.66s/batch, Batch Loss=0.774, Batch Tokens=64, Batch Time=5.814s]

