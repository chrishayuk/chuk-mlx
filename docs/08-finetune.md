
```bash
python finetune.py --config "./training_config/finetune/calvin-granite-3b-config.yaml"
```


# calvin infer
```bash
python infer.py --model "ibm-granite/granite-3b-code-instruct" --checkpoint "./output/checkpoints/checkpoint_epoch_1.npz"
```