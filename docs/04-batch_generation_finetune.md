# Introduction
These set of scripts give you the formula for creating finetune batches for training.

## Generate Fine Tune Batches
The following script will take a training jsonl file, tokenize it, and split into batches, ready for fine tuning.

TODO: DO A GRANITE VERSION and LLAMA 3 version

```bash
python batch_generator_finetune.py --input_files ./sample_data/calvin_scale_llama/train.jsonl --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output_directory ./output/batches/calvin --file_prefix calvin --max_sequence_length 4096 --batch_size 2
```

## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

### Input Batch
The following will analyze the input batch we created earlier

```bash
python tools/batch/batch_analyzer.py --batch_file output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2  
```

## Batch Viewer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

### Input Batch
The following will view the input batch we created earlier

```bash
python tools/batch/batch_viewer.py --batch_file output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/batch_viewer_cli.py --batch_file output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

## Npz Viewer
The following takes a npz file and performs analysis on the batch.
This will output the raw entries of the tensor

### Input Batch
The following will view the input batch we created earlier

```bash
python tools/batch/npz_viewer.py --batch_file output/batches/calvin/calvin_batch_0001.npz
```




