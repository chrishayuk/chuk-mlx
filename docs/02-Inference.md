# Introduction
infer.py is a ground up cli script that allows you to prompt a model using MLX.

## Inference
The following scripts show how to prompt a model

```bash
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
The following shows how to prompt the meta-llama/Meta-Llama-3-8B-Instruct model

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt "write a fibonacci function in python"
```

### Meta Llama 2
The following shows how to prompt the meta-llama/Llama-2-7b model

```bash
python infer.py --model meta-llama/Llama-2-7b --prompt "write a fibonacci function in python"
```

### IBM Granite
The following shows how to prompt the ibm/granite-7b-base model

```bash
python infer.py --model ibm/granite-7b-base --prompt "write a fibonacci function in python"
```

### IBM Granite Code
The following shows how to prompt the ibm/granite-7b-base model

```bash
python infer.py --model ibm-granite/granite-3b-code-instruct --prompt "write a fibonacci function in python"
```

### Mistral 7B
The following shows how to prompt the mistralai/Mistral-7B-Instruct-v0.2 model

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "write a fibonacci function in python"
```