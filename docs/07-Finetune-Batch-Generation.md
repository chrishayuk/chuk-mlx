```bash
python batch_generator_finetune.py --input_files ./sample_data/calvin_scale_llama/train.jsonl --tokenizer ibm-granite/granite-3b-code-instruct --output_directory ./output/calvin/batches --file_prefix calvin --max_sequence_length 512 --batch_size 16
```


## Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

The following will analyze the input batch we created earlier

```bash
python batch_analyzer.py --batch_file output/calvin/batches/calvin_batch_0001.npy --tokenizer ibm-granite/granite-3b-code-instruct
```

The following will analyze the target batch we created earlier

```bash
python batch_analyzer.py --batch_file output/calvin/batches/calvin_batch_0001_target.npy --tokenizer ibm-granite/granite-3b-code-instruct
```