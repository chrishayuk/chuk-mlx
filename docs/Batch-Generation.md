# Batches
This section describes howyou can pre-generate batching files for training the model.
This covers for testing purposes, pre-training purposes and fine tuning.

## Batch Generator - Pretrain
The follwoing script will take a jsonl dataset, tokenize it using the passed tokenizer, and split into batches, saving to the file system.

```bash
python batch_generator_pretrain.py --input_files ./sample_data/sample_training_data_small.jsonl --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output_directory ./output --file_prefix sample_tokenized --max_sequence_length 8096 --batch_size 1024
```

### LazyFox Batching
The following will batch up the lazy fox dataset:
```bash
python batch_generator_pretrain.py --input_files ./sample_data/lazyfox_train.jsonl --tokenizer lazyfox_tokenizer --output_directory ./output --file_prefix lazyfox --max_sequence_length 16 --batch_size 1024
```


### Batch Generator Pre-train Target Generator
The follwoing script will take a pre-train numpy file, generate a target file using a target shift, and save to the file system.  This gives you the next token generation files


```bash
python batch_pretrain_target_shift_generator.py --input_directory ./output --batch_prefix lazyfox
```
