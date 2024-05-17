# Introduction
infer.py is a ground up cli script that allows you to prompt a model using MLX.

## Inference
The following scripts show how to prompt a model

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

### Meta Llama 3
All Meta Llama 3 models are supported including:

- meta-llama/Meta-Llama-3-8B-Instruct

```python
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt "write a fibonacci function in python"
```

### Meta Llama 2
All Meta Llama 2 models are supported including:

- meta-llama/Llama-2-7b

```python
python infer.py --model meta-llama/Llama-2-7b --prompt "write a fibonacci function in python"
```

### IBM Granite
All IBM Granite models are supported including:

- ibm/granite-7b-base
- ibm-granite/granite-3b-code-instruct

```python
python infer.py --model ibm/granite-7b-base --prompt "write a fibonacci function in python"
```

### Mistral
The following MistralAI models are supported including:

- mistralai/Mistral-7B-Instruct-v0.2

```python
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "write a fibonacci function in python"
```