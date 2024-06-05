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
3 - fix checkpoint start and stopping
    - maybe a if there is no final model, start from checkpoint, unless overriden with flag to be clean
1 - Attempt Fine tuning of something like calvinscale
4 - validation datasets etc
6 - Revisit batching (bos,eos,pad), batch sizes and bucketing
5 - find optimal batch sizes
2 - refactor attention layer to attention block
5 - Attempt pretrain of something like tiny shakespeare, fineweb or tiny stories
1 - Models
    - Test new version of mistral
3 - Training
    - Implement LORA (model freezing, specifying layers)
    - Implement QLora (quantized)
2 - Modify model to support bloat16
1 - Generate large file generator for jsonl
1 - Memory Calculator
1 - attempt golden gate claude

Calvin Run (no lora, 3b code)
402/403 [37:56<00:05,  5.66s/batch, Batch Loss=0.774, Batch Tokens=64, Batch Time=5.814s]

