# Introduction
These set of scripts give you the formula for creating finetune batches for training.

## Generate Fine Tune Batches
The following script will take a training jsonl file, tokenize it, and split into batches, ready for fine tuning.

```bash
python batch_generator_finetune.py --input_files ./sample_data/calvin_scale_llama/train.jsonl --tokenizer ibm-granite/granite-3b-code-instruct --output_directory ./output/calvin/batches --file_prefix calvin --max_sequence_length 100 --batch_size 16
```

## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

### Input Batch
The following will analyze the input batch we created earlier

```bash
python batch_analyzer.py --batch_file output/calvin/batches/calvin_batch_0001.npy --tokenizer ibm-granite/granite-3b-code-instruct
```

### Target Batch
The following will analyze the input batch we created earlier

```bash
python batch_analyzer.py --batch_file output/calvin/batches/calvin_batch_0001_target.npy --tokenizer ibm-granite/granite-3b-code-instruct
```

## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

### Input Batch
The following will view the input batch we created earlier

```bash
python batch_viewer.py --batch_file output/calvin/batches/calvin_batch_0001.npy --tokenizer ibm-granite/granite-3b-code-instruct
```

### Target Batch
The following will view the input batch we created earlier

```bash
python batch_viewer.py --batch_file output/calvin/batches/calvin_batch_0001_target.npy --tokenizer ibm-granite/granite-3b-code-instruct
```

## Notes

### Max Sequence Length 100 (padded) - Batch Size of 16
Epoch [1/1]:   5%|▍        | 20/403 [01:15<24:28,  3.83s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=3.887s, Tokens/s=60.97, LR=0.0000200]2024-05-19 18:56:11,681 - INFO - Epoch [1/1] completed. Loss: 0.2199
2024-05-19 18:56:11,681 - INFO - Checkpointing at end of epoch 1
2024-05-19 18:56:14,064 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 18:56:14,064 - INFO - Total training time: 75.248s
2024-05-19 18:56:14,064 - INFO - Total iterations: 20
2024-05-19 18:56:14,064 - INFO - Average batch time: 3.762s
2024-05-19 18:56:14,064 - INFO - Tokens per second: 62.99 (Actual) / 425.26 (Theoretical)
Epoch [1/1]:   5%|▍        | 20/403 [01:17<24:46,  3.88s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=3.887s, Tokens/s=60.97, LR=0.0000200]

### Max Sequence Length 100 (padded) - Batch Size of 32
Epoch [1/1]:   5%|▍        | 20/403 [02:14<44:03,  6.90s/batch, Batch Loss=2.88, Batch Tokens=428, Batch Time=6.931s, Tokens/s=61.75, LR=0.0000200]2024-05-19 20:11:51,438 - INFO - Epoch [1/1] completed. Loss: 0.2261
2024-05-19 20:11:51,438 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:11:53,953 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:11:53,953 - INFO - Total training time: 134.814s
2024-05-19 20:11:53,953 - INFO - Total iterations: 20
2024-05-19 20:11:53,953 - INFO - Average batch time: 6.740s
2024-05-19 20:11:53,953 - INFO - Tokens per second: 63.49 (Actual) / 474.73 (Theoretical)
Epoch [1/1]:   5%|▍        | 20/403 [02:17<43:49,  6.87s/batch, Batch Loss=2.88, Batch Tokens=428, Batch Time=6.931s, Tokens/s=61.75, LR=0.0000200]

This is telling us that, theoretically that batch loading is so efficient, it doesn't make much difference on the batch size...

### Max Sequence Length 100 (padded) - Batch Size of 8
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:   5%|▍        | 20/403 [00:46<14:42,  2.30s/batch, Batch Loss=2.44, Batch Tokens=132, Batch Time=2.324s, Tokens/s=56.79, LR=0.0000200]2024-05-19 20:15:55,695 - INFO - Epoch [1/1] completed. Loss: 0.2246
2024-05-19 20:15:55,695 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:15:58,169 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:15:58,170 - INFO - Total training time: 46.323s
2024-05-19 20:15:58,170 - INFO - Total iterations: 20
2024-05-19 20:15:58,170 - INFO - Average batch time: 2.316s
2024-05-19 20:15:58,170 - INFO - Tokens per second: 56.99 (Actual) / 345.40 (Theoretical)
Epoch [1/1]:   5%|▍        | 20/403 [00:48<15:34,  2.44s/batch, Batch Loss=2.44, Batch Tokens=132, Batch Time=2.324s, Tokens/s=56.79, LR=0.0000200]

This is telling us that we're taking a small hit for batch loading with a smaller batch.

### Max Sequence Length 100 (padded) - Batch Size of 4
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:   5%|▍         | 20/403 [00:32<10:12,  1.60s/batch, Batch Loss=2.39, Batch Tokens=64, Batch Time=1.594s, Tokens/s=40.16, LR=0.0000200]2024-05-19 20:17:52,962 - INFO - Epoch [1/1] completed. Loss: 0.2687
2024-05-19 20:17:52,963 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:17:55,484 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:17:55,484 - INFO - Total training time: 32.401s
2024-05-19 20:17:55,484 - INFO - Total iterations: 20
2024-05-19 20:17:55,484 - INFO - Average batch time: 1.620s
2024-05-19 20:17:55,484 - INFO - Tokens per second: 39.51 (Actual) / 246.91 (Theoretical)
Epoch [1/1]:   5%|▍         | 20/403 [00:34<11:08,  1.75s/batch, Batch Loss=2.39, Batch Tokens=64, Batch Time=1.594s, Tokens/s=40.16, LR=0.0000200]

Now we're taking a hit on the tokens per second..

There is probably a case for a batch size optimizer... focusing on tokens per second..


### Max Sequence Length 500 (padded) - Batch Size of 16
Epoch [1/1]:   5%|▎      | 20/403 [06:38<2:41:12, 25.25s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=24.811s, Tokens/s=9.55, LR=0.0000200]2024-05-19 19:05:08,882 - INFO - Epoch [1/1] completed. Loss: 0.2199
2024-05-19 19:05:08,883 - INFO - Checkpointing at end of epoch 1
2024-05-19 19:05:11,249 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 19:05:11,249 - INFO - Total training time: 398.338s
2024-05-19 19:05:11,249 - INFO - Total iterations: 20
2024-05-19 19:05:11,249 - INFO - Average batch time: 19.917s
2024-05-19 19:05:11,249 - INFO - Tokens per second: 11.90 (Actual) / 411.31 (Theoretical)
Epoch [1/1]:   5%|▎      | 20/403 [06:40<2:07:53, 20.04s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=24.811s, Tokens/s=9.55, LR=0.0000200]

### Max Sequence Length 20 (padded) - Batch Size of 16
Epoch [1/1]:   5%|▎      | 20/403 [06:38<2:41:12, 25.25s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=24.811s, Tokens/s=9.55, LR=0.0000200]2024-05-19 19:05:08,882 - INFO - Epoch [1/1] completed. Loss: 0.2199
2024-05-19 19:05:08,883 - INFO - Checkpointing at end of epoch 1
2024-05-19 19:05:11,249 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 19:05:11,249 - INFO - Total training time: 398.338s
2024-05-19 19:05:11,249 - INFO - Total iterations: 20
2024-05-19 19:05:11,249 - INFO - Average batch time: 19.917s
2024-05-19 19:05:11,249 - INFO - Tokens per second: 11.90 (Actual) / 411.31 (Theoretical)
Epoch [1/1]:   5%|▎      | 20/403 [06:40<2:07:53, 20.04s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=24.811s, Tokens/s=9.55, LR=0.0000200]

### Max Sequence Length 20 (padded) - Batch Size of 16
Epoch [1/1]:   5%|▍       | 20/403 [00:28<08:52,  1.39s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=1.393s, Tokens/s=170.16, LR=0.0000200]2024-05-19 20:20:15,445 - INFO - Epoch [1/1] completed. Loss: 0.2199
2024-05-19 20:20:15,445 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:20:17,942 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:20:17,942 - INFO - Total training time: 28.185s
2024-05-19 20:20:17,942 - INFO - Total iterations: 20
2024-05-19 20:20:17,942 - INFO - Average batch time: 1.409s
2024-05-19 20:20:17,942 - INFO - Tokens per second: 168.18 (Actual) / 227.07 (Theoretical)
Epoch [1/1]:   5%|▍       | 20/403 [00:30<09:47,  1.53s/batch, Batch Loss=2.59, Batch Tokens=237, Batch Time=1.393s, Tokens/s=170.16, LR=0.0000200]

This shows, the power of getting the padding sorted..
Tokens per second pushes upto 170.16

### Max Sequence Length 20 (padded) - Batch Size of 24
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:   5%|▍       | 20/403 [00:34<10:52,  1.70s/batch, Batch Loss=2.58, Batch Tokens=330, Batch Time=1.704s, Tokens/s=193.66, LR=0.0000200]2024-05-19 20:22:46,045 - INFO - Epoch [1/1] completed. Loss: 0.2223
2024-05-19 20:22:46,045 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:22:48,547 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:22:48,547 - INFO - Total training time: 34.318s
2024-05-19 20:22:48,547 - INFO - Total iterations: 20
2024-05-19 20:22:48,548 - INFO - Average batch time: 1.716s
2024-05-19 20:22:48,548 - INFO - Tokens per second: 192.32 (Actual) / 279.73 (Theoretical)
Epoch [1/1]:   5%|▍       | 20/403 [00:36<11:45,  1.84s/batch, Batch Loss=2.58, Batch Tokens=330, Batch Time=1.704s, Tokens/s=193.66, LR=0.0000200]

### Max Sequence Length 20 (padded) - Batch Size of 32
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:   5%|▍       | 20/403 [00:39<12:50,  2.01s/batch, Batch Loss=2.88, Batch Tokens=428, Batch Time=1.992s, Tokens/s=214.87, LR=0.0000200]2024-05-19 20:24:03,954 - INFO - Epoch [1/1] completed. Loss: 0.2261
2024-05-19 20:24:03,954 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:24:06,453 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:24:06,453 - INFO - Total training time: 39.840s
2024-05-19 20:24:06,453 - INFO - Total iterations: 20
2024-05-19 20:24:06,453 - INFO - Average batch time: 1.992s
2024-05-19 20:24:06,453 - INFO - Tokens per second: 214.86 (Actual) / 321.29 (Theoretical)
Epoch [1/1]:   5%|▍       | 20/403 [00:42<13:30,  2.12s/batch, Batch Loss=2.88, Batch Tokens=428, Batch Time=1.992s, Tokens/s=214.87, LR=0.0000200]

### Max Sequence Length 20 (padded) - Batch Size of 48
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:   5%|▍       | 20/403 [01:04<21:47,  3.41s/batch, Batch Loss=2.63, Batch Tokens=794, Batch Time=3.423s, Tokens/s=231.93, LR=0.0000200]2024-05-19 20:25:50,133 - INFO - Epoch [1/1] completed. Loss: 0.2291
2024-05-19 20:25:50,133 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:25:52,157 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:25:52,158 - INFO - Total training time: 64.982s
2024-05-19 20:25:52,158 - INFO - Total iterations: 20
2024-05-19 20:25:52,158 - INFO - Average batch time: 3.249s
2024-05-19 20:25:52,158 - INFO - Tokens per second: 244.38 (Actual) / 393.96 (Theoretical)
Epoch [1/1]:   5%|▍       | 20/403 [01:07<21:23,  3.35s/batch, Batch Loss=2.63, Batch Tokens=794, Batch Time=3.423s, Tokens/s=231.93, LR=0.0000200]


### Max Sequence Length 20 (padded) - Batch Size of 64
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:  77%|██████▉  | 20/26 [01:03<00:18,  3.16s/batch, Batch Loss=2.63, Batch Tokens=794, Batch Time=3.164s, Tokens/s=250.98, LR=0.0000200]2024-05-19 20:39:11,854 - INFO - Epoch [1/1] completed. Loss: 3.5517
2024-05-19 20:39:11,854 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:39:14,313 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:39:14,313 - INFO - Total training time: 63.904s
2024-05-19 20:39:14,313 - INFO - Total iterations: 20
2024-05-19 20:39:14,313 - INFO - Average batch time: 3.195s
2024-05-19 20:39:14,313 - INFO - Tokens per second: 248.50 (Actual) / 400.60 (Theoretical)
Epoch [1/1]:  77%|██████▉  | 20/26 [01:06<00:19,  3.32s/batch, Batch Loss=2.63, Batch Tokens=794, Batch Time=3.164s, Tokens/s=250.98, LR=0.0000200]


### Max Sequence Length 20 (padded) - Batch Size of 128
Model Loaded: ibm-granite/granite-3b-code-instruct
loading weights
weights loaded
Epoch [1/1]:  92%|███████▍| 12/13 [01:11<00:05,  5.83s/batch, Batch Loss=3.64, Batch Tokens=1529, Batch Time=5.739s, Tokens/s=266.44, LR=0.0000200]2024-05-19 20:37:29,589 - ERROR - Error in batch 13: Shapes (75,20) and (128,20) cannot be broadcast.
2024-05-19 20:37:29,590 - INFO - Epoch [1/1] completed. Loss: 5.3410
2024-05-19 20:37:29,590 - INFO - Checkpointing at end of epoch 1
2024-05-19 20:37:32,112 - INFO - Saved checkpoint: ./output/calvin/checkpoints/checkpoint_epoch_1.npz
2024-05-19 20:37:32,112 - INFO - Total training time: 71.769s
2024-05-19 20:37:32,112 - INFO - Total iterations: 12
2024-05-19 20:37:32,112 - INFO - Average batch time: 5.980s
2024-05-19 20:37:32,112 - INFO - Tokens per second: 255.65 (Actual) / 428.04 (Theoretical)
Epoch [1/1]:  92%|███████▍| 12/13 [01:14<00:06,  6.19s/batch, Batch Loss=3.64, Batch Tokens=1529, Batch Time=5.739s, Tokens/s=266.44, LR=0.0000200]




