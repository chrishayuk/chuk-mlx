# Introduction
This is a ground up rework of an MLX training script.

## Supported Models
The following model familes are supported.

- Meta Llama-3
- Meta Llama-2
- Mistral-7B
- IBM Granite
- IBM Granite Code Models



### Meta Llama 3
All Meta Llama 3 models are supported including:

- meta-llama/Meta-Llama-3-8B-Instruct

### Meta Llama 2
All Meta Llama 2 models are supported including:

- meta-llama/Llama-2-7b

### IBM Granite
All IBM Granite models are supported including:

- ibm/granite-7b-base
- ibm-granite/granite-3b-code-instruct

### Mistral
The following MistralAI models are supported including:

- mistralai/Mistral-7B-Instruct-v0.2

## Infer

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2
```

```bash
python infer.py --model ibm/granite-7b-base
```

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

TODO
1 - Refactor train.py, to move inference out into infer
1 - Generate large file generator for jsonl
2 - Batch Sequence Visualizer (batch name, row)
3 - Migrate trainer to use mx data loader
4 - Migrate lazy fox to use data loader
