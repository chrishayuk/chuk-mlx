# Diagnosis tools
As it stands there is a set of utiltiies that help you benchmark and diagnose the performance of MLX.

## Batch Loader Test
The following script performs a batch load of the batches into memory.
It also produces a pretty summary table that gives the loading times etc

```bash
python batch_loader_test.py --output_directory ./output --file_prefix sample
```

## Batch Memory Calculator
The following script calculates the memory requirements per batch and across a number of batches.
This allows us to calculate how many batches we could load at once.
```bash
python batch_memory_calculator.py --max_sequence_length 4096 --batch_size 1024 --dtype float32 --num_batches 1024
```

## Batch Token Calculator
The following script calculates the the number of batches required for a given number of tokens.
This allows us to calculate how many batches we could load at once.

```bash
python batch_token_calculator.py --max_sequence_length 8096 --batch_size 1024 --dtype float32 --num_tokens 1024
```

## Batch Generator - Random Numpy Batches
The following script generates random numpy batches shaped to match the model.
This contains nonsense data but can be used to performance test batch loading.
This will produce a table outlining the generation performance times.

The following generates a llama-3 style batch
```bash
python batch_generator_random.py --vocab_size 128256 --max_sequence_length 8096 --batch_size 1024 --num_batches 1 --file_prefix llama3
```

## Batch Analyzer - Random Numpy Batches
This analyses as fine tuned batch input:

```bash
python batch_analyzer.py --batch_file output/calvin_batch_0001.npy --tokenizer meta-llama/Meta-Llama-3-8B-Instruct
```