# Fine tuning with original mlx (lora)
In order to make sure that our pipeline can fine tune, we need some test code to ensure everything is working as expected.

## Inferring with Lora
I have included a fine tuned tiny-llama adapters file that has been trained using the fine tune command shown later.  This adapters file has been trained on 8 lora layers, for 1000 iterations, with a batch size of 2 on the calvin scale llama dataset.

It's not the best dataset in the world and it's overfitted and trashed the model but it proves the fine tune works.

here is the command you can use to test the infer within the mlx-examples/lora folder.  later on, i'll provide the command to test the same file with out inference code.

```bash
python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file /Users/christopherhay/chris-source/chuk-mlx/sample_data/calvin_scale_llama/lora-test.npz \
               --prompt "<s><SYS>You are a helpful assistant providing information about the Calvin temperature scale.</SYS>[INST]How is 45 Calvin usually categorized?[/INST]"
```

and

```bash
python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file /Users/christopherhay/chris-source/chuk-mlx/sample_data/calvin_scale_llama/lora-test.npz \
               --prompt "<s>[INST]How would you describe 5 degrees calvin?[/INST]"
```

and

```bash
python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file /Users/christopherhay/chris-source/chuk-mlx/sample_data/calvin_scale_llama/lora-test.npz \
               --prompt "<s>[INST]Who is Ada Lovelace?[/INST]"
```

## Fine tuning with Lora
This allows you to perform the same fine tune that i created earlier with lora.py script in the mlx-examples/lora folder.  This will fine tune a tinyllama model using the calvinscale dataset with a batch size of 2, 8 lora layers, for 1000 iterations.

```bash
python lora.py --train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data /Users/christopherhay/chris-source/chuk-datasets/datasets/sample/output/llama/ --batch-size 2 --lora-layers 8 --iters 1000
```

python lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
               --adapter-file ./adapters.npz \
               --prompt "<s>[INST]Who is Ada Lovelace?[/INST]"