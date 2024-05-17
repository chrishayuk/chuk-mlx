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
1 - Modify infer to be interactive
2 - Clean up the inference utilities
2 - Modify model to support bloat16
1 - Generate large file generator for jsonl
2 - Batch Sequence Visualizer (batch name, row)
3 - Migrate trainer to use mx data loader
4 - Migrate lazy fox to use data loader
5 - Inference performance benchmarking tool that outputs a table of tokens per second
6 - Implement support for quantization

