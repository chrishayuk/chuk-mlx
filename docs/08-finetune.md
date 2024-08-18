
```bash
    python train.py --config "./training_config/finetune/calvin-granite-3b-config.yaml"
```

```bash
python train.py --config "./training_config/finetune/calvin-mistral-7b-config.yaml"
```

```bash
python train.py --config "./training_config/finetune/sample-tinyllama-config.yaml"
```

```bash
python train.py --config "./training_config/finetune/calvin-tinyllama-3b-config.yaml"
```

# calvin infer


### granite-7b
```bash
python infer.py --model "ibm/granite-7b-instruct" --checkpoint "./output/calvin/checkpoints/checkpoint_epoch_1.npz"
```

```bash
python infer.py --model "ibm-granite/granite-3b-code-instruct" --checkpoint "./output/sample_llama/checkpoints/final_model_checkpoint.npz"

### granite-3b
```bash
python infer.py --model "ibm-granite/granite-3b-code-instruct" --checkpoint "./output/calvin/checkpoints/checkpoint_epoch_1.npz"
```

```bash
python infer.py --model "ibm-granite/granite-3b-code-instruct" --checkpoint "./output/sample_llama/checkpoints/final_model_checkpoint.npz"
```

### mistral-7b-0.2
```bash
python infer.py --model "mistralai/Mistral-7B-Instruct-v0.2" --checkpoint "./output/calvin/checkpoints/final_model_checkpoint.npz"
```

### tiny llama
```bash
python infer.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --checkpoint "./output/calvin/checkpoints/final_model_checkpoint.npz"
```
```bash
python infer.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --checkpoint "./output/calvin/checkpoints/checkpoint_epoch_13.npz"
```

use the lora version

```bash
python infer.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --checkpoint "./output/sample/checkpoints/final_model_checkpoint.npz"
```

python lora.py --train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data /Users/christopherhay/chris-source/chuk-datasets/datasets/calvin_scale/output/llama/ --batch-size 2 --lora-layers 8 --iters 1000

python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file ./adapters.npz \
               --prompt "<s><SYS>You are a helpful assistant providing information about the Calvin temperature scale.</SYS>[INST]How is -45 Calvin usually categorized?[/INST]"

python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file ./adapters.npz \
               --prompt "<s>[INST]How is -45 Calvin usually categorized?[/INST]"


# TESTING WITH LORA

```bash
python lora.py --train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data /Users/christopherhay/chris-source/chuk-datasets/datasets/sample/output/llama/ --batch-size 2 --lora-layers 8 --iters 1000
```

python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file ./adapters.npz \
               --prompt "USER: Who is Ada Lovelace? ASSISTANT: "

python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file ./adapters.npz \
               --prompt "Q: What festival is celebrated on Zelara?  A: "
