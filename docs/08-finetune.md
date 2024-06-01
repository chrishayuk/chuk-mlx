
```bash
    python train.py --config "./training_config/finetune/calvin-granite-3b-config.yaml"
```

```bash
python train.py --config "./training_config/finetune/calvin-mistral-7b-config.yaml"
```


# calvin infer



### granite-3b
```bash
python infer.py --model "ibm-granite/granite-3b-code-instruct" --checkpoint "./output/calvin/checkpoints/checkpoint_epoch_1.npz"
```

### mistral-7b-0.2
```bash
python infer.py --model "mistralai/Mistral-7B-Instruct-v0.2" --checkpoint "./output/calvin/checkpoints/final_model_checkpoint.npz"
```

use the lora version

```bash
python infer.py --model "mistralai/Mistral-7B-Instruct-v0.2" --checkpoint "/Users/christopherhay/chris-source/mlx-examples/lora/adapters.npz"
```