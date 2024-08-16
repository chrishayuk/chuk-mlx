# Pretrain Batch Generation
This section describes how you can pre-generate batching files for pre-training the model.

For context, in order to speed up training of models with chuk-mlx.  We pre-process the datasets, this means the trainer is not performing any tokenization or batching during the pre-training process.  This means it is purely loading batches and processing.

## Generating Batchfiles for inputs (next token prediction)
The following script will take a jsonl dataset, tokenize it using the passed tokenizer, and split into batches, saving to the file system.

```bash
python batch_generator_pretrain.py --input_files ./sample_data/sample_training_data_small.jsonl --tokenizer ibm-granite/granite-3b-code-instruct --output_directory ./output/sample/batches --file_prefix sample --max_sequence_length 8096 --batch_size 1024
```

## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

The following will analyze the input batch we created earlier

```bash
python batch_analyzer.py --batch_file output/sample/batches/sample_batch_0001.npz --tokenizer ibm-granite/granite-3b-code-instruct
```

## Batch Viewer
The following takes a batch file and allows you to the view the batch.

The following will view the input batch we created earlier

```bash
python batch_viewer.py --batch_file output/sample/batches/sample_batch_0001.npz --tokenizer ibm-granite/granite-3b-code-instruct
```

