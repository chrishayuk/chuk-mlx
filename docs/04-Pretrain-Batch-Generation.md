# Pretrain Batch Generation
This section describes how you can pre-generate batching files for pre-training the model.

For context, in order to speed up training of models with chuk-mlx.  We pre-process the datasets, this means the trainer is not performing any tokenization or batching during the pre-training process.  This means it is purely loading batches and processing.

There are 2 steps in the process currently

1. Genenerating the pretrain input batchfiles
2. Generating the the pretrain target batchfiles for each input batchfile.

The target batchfile is a next token shift of the batchfile.

## Generating Batchfiles for inputs
The following script will take a jsonl dataset, tokenize it using the passed tokenizer, and split into batches, saving to the file system.

```bash
python batch_generator_pretrain.py --input_files ./sample_data/sample_training_data_small.jsonl --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output_directory ./output/sample/batches --file_prefix sample --max_sequence_length 8096 --batch_size 1024
```

## Generating Batchfiles for targets (next token prediction)
The following script will take a pre-train numpy file, generate a target file using a target shift, and save to the file system.  This gives you the next token generation files

```bash
python batch_pretrain_target_shift_generator.py --input_directory ./output/sample/batches --batch_prefix sample
```

## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

The following will analyze the input batch we created earlier

```bash
python batch_analyzer.py --batch_file output/sample/batches/sample_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

## Batch Viewer
The following takes a batch file and allows you to the view the batch.

The following will view the input batch we created earlier

```bash
python batch_viewer.py --batch_file output/sample/batches/sample_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

