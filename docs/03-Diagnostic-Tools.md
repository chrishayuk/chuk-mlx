# Diagnosis tools
As it stands there is a set of utiltiies that help you benchmark and diagnose the performance of MLX.

## Model Viewer - Show Layers
This is a simple utility that allows you to print the configuration of a model

```bash
python model_viewer.py --model ibm-granite/granite-3b-code-instruct --show-config
```

## Model Viewer - Show Config
This is a simple utility that allows you to print the configuration of a model

```bash
#python model_viewer.py --model mistralai/Mistral-7B-Instruct-v0.2 --show-layers
python model_viewer.py --model ibm-granite/granite-3b-code-instruct --show-layers
```

## Print Layer Modules
This is a simple utility that allows you to print the modules of a layer.

```bash
#python print_layer_modules.py --model ibm/granite-7b-base
python print_layer_modules.py --model ibm-granite/granite-3b-code-instruct
```
    
## Print Tokens
This is a simple utility that allows you to print tokens for the given tokenizer

```bash
#python print_tokens.py --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --prompt "Who is Ada Lovelace?"
python print_tokens.py --tokenizer ibm-granite/granite-3b-code-instruct --prompt "Who is Ada Lovelace?"
```
