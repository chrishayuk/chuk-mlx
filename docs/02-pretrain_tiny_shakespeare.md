# Introduction
This is a simple script that allows us to test pre-training works.

## Dataset
The tiny shakespeare dataset is in:

- sample_data/datasets/tiny_shakespeare.txt

## Pre-Processing
Before training on the tinyshakespeare dataset, we need to pre-process the data to put into a standard jsonl form.
While, we're at it, we need to break up the text so it will fit within the max sequence length.
For pretrain, we'll have our max_sequence_length set to 4096
To pre-process run the following

```bash
python text_to_jsonl_preprocessor.py
```

This will place the jsonl version of the file into

- output/datasets/tiny_shakespeare.jsonl

## Batching
We then need to split the dataset into batches for trainimg

```bash
python batch_generator_pretrain.py --input_files output/datasets/tiny_shakespeare.jsonl --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output_directory ./output/batches/tiny_shakespeare --file_prefix tiny_shakespeare --max_sequence_length 4096 --batch_size 64 --regenerate_batches
```

This will place the batches in

- output/batches/tiny_shakespeare

It should generate 5 batches

### Batch Analysis

```bash
python tools/batch/batch_analyzer.py --batch_file output/batches/tiny_shakespeare/tiny_shakespeare_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

### Batch Viewer

```bash
python tools/batch/batch_viewer.py --batch_file output/batches/tiny_shakespeare/tiny_shakespeare_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/batch_viewer_cli.py --batch_file output/batches/tiny_shakespeare/tiny_shakespeare_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```