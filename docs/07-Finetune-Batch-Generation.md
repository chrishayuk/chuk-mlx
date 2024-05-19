# Introduction
These set of scripts give you the formula for creating finetune batches for training.

## Generate Fine Tune Batches
The following script will take a training jsonl file, tokenize it, and split into batches, ready for fine tuning.

```bash
python batch_generator_finetune.py --input_files ./sample_data/calvin_scale_llama/train.jsonl --tokenizer ibm-granite/granite-3b-code-instruct --output_directory ./output/calvin/batches --file_prefix calvin --max_sequence_length 512 --batch_size 16
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
Before Update
otal training time: 0.104s [00:00<00:00, 460.41batch/s, Batch Loss=0.148, Batch Tokens=528, Batch Time=0.002s, Tokens/s=253821.49 (Act
Total iterations: 50
Average batch time: 0.002s
Tokens per second: 253388.77 (Actual) / 253388.77 (Theoretical)
Epoch [50/50]: 100%|█| 1/1 [00:00<00:00, 444.69batch/s, Batch Loss=0.148, Batch Tokens=528, Batch Time=0.002s, Tokens/s=253821.49 (Act


Completed Training

Input sequence: the quick brown
Predicted next word: fox

Post Update

024-05-19 14:09:38,106 - INFO - Total training time: 0.103sch Loss=0.143, Batch Tokens=528, Batch Time=0.001s, Tokens/s=426785.99 (Act
2024-05-19 14:09:38,106 - INFO - Total iterations: 50
2024-05-19 14:09:38,106 - INFO - Average batch time: 0.002s
2024-05-19 14:09:38,106 - INFO - Tokens per second: 256317.98 (Actual) / 256317.98 (Theoretical)
Epoch [50/50]: 100%|█| 1/1 [00:00<00:00, 608.58batch/s, Batch Loss=0.143, Batch Tokens=528, Batch Time=0.001s, Tokens/s=426785.99 (Act


Completed Training

Input sequence: the quick brown
Predicted next word: fox


Epoch [50/50]:   0%|                                                                                         | 0/1 [00:00<?, ?batch/s2024-05-19 14:12:29,635 - INFO - Total training time: 0.103sch Loss=0.146, Batch Tokens=528, Batch Time=0.001s, Tokens/s=419033.59 (Act
2024-05-19 14:12:29,635 - INFO - Total iterations: 50
2024-05-19 14:12:29,635 - INFO - Average batch time: 0.002s
2024-05-19 14:12:29,635 - INFO - Tokens per second: 255798.10 (Actual) / 255798.10 (Theoretical)
Epoch [50/50]: 100%|█| 1/1 [00:00<00:00, 611.59batch/s, Batch Loss=0.146, Batch Tokens=528, Batch Time=0.001s, Tokens/s=419033.59 (Act

Epoch [1/1]:   1%| | 5/403 [00:22<29:58,  4.52s/batch, Batch Loss=11.6, Batch Tokens=64, Batch Time=4.508s, Tokens/s=14.20 (Actual) / 2024-05-19 14:16:07,932 - INFO - Epoch [1/1] completed. Loss: 0.1133
2024-05-19 14:16:07,932 - INFO - Total training time: 22.678s
2024-05-19 14:16:07,932 - INFO - Total iterations: 5
2024-05-19 14:16:07,932 - INFO - Average batch time: 4.535s
2024-05-19 14:16:07,932 - INFO - Tokens per second: 14.11 (Actual) / 451.53 (Theoretical)
Epoch [1/1]:   1%| | 5/403 [00:22<30:05,  4.54s/batch, Batch Loss=11.6, Batch Tokens=64, Batch Time=4.508s, Tokens/s=14.20 (Actual) / 
