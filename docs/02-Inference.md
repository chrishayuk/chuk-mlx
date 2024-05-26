# Introduction
infer.py is a ground up cli script that allows you to prompt a model using MLX.

## Interactive Chat (with Chat Template)
The following shows how to have an interactive chat with a chat model

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Text Generation - Passing a prompt
The following shows how to pass a prompt using a chat template

```bash
python infer.py --model ibm/granite-7b-base --prompt "who is ada lovelace?"
```

## Text Generation - Passing a prompt (with Chat Template)
The following shows how to pass a prompt using a chat template

```bash
python infer.py --model meta-llama/Llama-2-7b-chat-hf --prompt "Who is Ada Lovelace" --chat
```