TODO
1 - Generate large file generator for jsonl
2 - Batch Sequence Visualizer (batch name, row)
3 - Migrate trainer to use mx data loader
4 - Migrate lazy fox to use data loader

# Introduction
This is a ground up rework of an MLX training script.

## Diagnosis tools
As it stands there is a set of utiltiies that help you benchmark and diagnose the performance of MLX.

## Batch Loader


## Model Downloader
This is a simple utility that allows you to download a model from huggingface hug, and stick it the cache

```bash
python model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.2
```

## Batches

### Batch Generator - Random Numpy Batches
The following script generates random numpy batches shaped to match the model.
This contains nonsense data but can be used to performance test batch loading.
This will produce a table outlining the generation performance times.

The following generates a llama-3 style batch
```bash
python batch_generator_random.py --vocab_size 128256 --max_sequence_length 8096 --batch_size 1024 --num_batches 1 --file_prefix llama3
```

### Batch Generator
The follwoing script will take a jsonl dataset, tokenize it using the passed tokenizer, and split into batches, saving to the file system.

```bash
python batch_generator.py --input_files ./sample_data/sample_training_data_small.jsonl --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output_directory ./output --file_prefix sample_tokenized --max_sequence_length 8096 --batch_size 1024
```

#### LazyFox Batching
The following will batch up the lazy fox dataset:
```bash
python batch_generator.py --input_files ./sample_data/lazyfox_train.jsonl --tokenizer lazyfox_tokenizer --output_directory ./output --file_prefix lazyfox --max_sequence_length 16 --batch_size 1024
```

### Batch Analyzer
The following takes a batch file and performs analysis on the batch.
This will produce a pretty table that outlines rows, tokens per batch, padding tokens etc

```bash
python batch_analyzer.py --batch_file output/sample_tokenized_batch_0001.npy --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

#### LazyFox Analysis
The following will batch up the lazy fox dataset:
```bash
python batch_analyzer.py --batch_file output/lazyfox_batch_0001.npy --tokenizer lazyfox_tokenizer
```

### Batch Memory Calculator
The following script calculates the memory requirements per batch and across a number of batches.
This allows us to calculate how many batches we could load at once.
```bash
python batch_memory_calculator.py --max_sequence_length 4096 --batch_size 1024 --dtype float32 --num_batches 1024
```

### Batch Token Calculator
The following script calculates the the number of batches required for a given number of tokens.
This allows us to calculate how many batches we could load at once.

```bash
python batch_token_calculator.py --max_sequence_length 8096 --batch_size 1024 --dtype float32 --num_tokens 1024
```

### Batch Loader Test
The following script performs a batch load of the batches into memory.
It also produces a pretty summary table that gives the loading times etc

```bash
python batch_token_calculator.py --max_sequence_length 8096 --batch_size 1024 --dtype float32 --num_tokens 1024
```

## Model Viewer
This is a simple utility that allows you to view a models config or layers.

```bash
python model_viewer.py --model meta-llama/Meta-Llama-3-8B-Instruct --show-config
```

or

```bash
python model_viewer.py --model ibm/granite-7b-base --show-layers
```

## Print Layer Modules
This is a simple utility that allows you to print the modules of a layer.

```bash
python print_layer_modules.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

or

```bash
python print_layer_modules.py --model ibm/granite-7b-base
```
    
## Print Tokens
This is a simple utility that allows you to print tokens for the given tokenizer

```bash
python print_tokens.py --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --prompt "Who is Ada Lovelace?"
```

or

```bash
python print_tokens.py --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --prompt "Who is Ada Lovelace?"
```

or

```bash
python print_tokens.py --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --prompt "the quick brown fox jumps over the lazy dog"
```

or

```bash
python print_tokens.py --tokenizer ibm/granite-7b-base --prompt "the quick brown fox jumps over the lazy dog"
```

or

```bash
python print_tokens.py --tokenizer lazyfox_tokenizer --prompt "the quick brown fox jumps over the lazy dog"
```

## Infer

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2
```

```bash
python infer.py --model ibm/granite-7b-base
```

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct
```