# Introduction
infer.py is a ground up cli script that allows you to prompt a model using MLX.

## Inference
The following scripts show how to prompt a model

## Llama Models
The following shows how to interact with the Meta Llama 2 and 3 models.

### Meta Llama 3
The following shows how to prompt the meta-llama/Meta-Llama-3-8B-Instruct model

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt "write a fibonacci function in python"
```

and chat mode

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Meta Llama 2
The following shows how to prompt the meta-llama/Llama-2-7b model

```bash
python infer.py --model meta-llama/Llama-2-7b --prompt "write a fibonacci function in python"
```

or for chat

```bash
python infer.py --model meta-llama/Llama-2-7b-chat-hf
```

## Llama Models
The following shows how to interact with the Mistral Models

### Mistral 7B
The following shows how to prompt the mistralai/Mistral-7B-Instruct-v0.2 model

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "write a fibonacci function in python"
```

or chat

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2
```


## IBM Models
The following shows how to interact with the IBM Granite and Granite Code Models

### Granite 3B Code Instruct

```bash
python infer.py --model ibm-granite/granite-3b-code-instruct --prompt "write a fibonacci function in python"
```

### IBM Granite
The following shows how to prompt the ibm/granite-7b-base model

```bash
python infer.py --model ibm/granite-7b-base --prompt "who is ada lovelace?"
```

### IBM Granite Code
The following shows how to prompt the ibm/granite-7b-base model

```bash
python infer.py --model ibm-granite/granite-3b-code-instruct --prompt "write a fibonacci function in python"
```

### IBM Merlinite
The following shows how to prompt the ibm/merlinite-7b model

```bash
python infer.py --model ibm/merlinite-7b --prompt "write a fibonacci function in python"
```