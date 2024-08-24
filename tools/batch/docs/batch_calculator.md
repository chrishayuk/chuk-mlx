# Batch Calculators
A set of calculators to help you calculate batch tokens or memory.

## Batch Token Calculator
This is a pretty useful utility where you can provide the following information

- batch size
- data type
- max sequence length
- number of sequences (rows) per batch
- number of tokens

So for example, to calculate the number of batches (and size of disk) for 1 trillion tokens in a batch of 1024, with a max sequence length of 8192.  you can run the following command


```bash
python tools/batch/batch_token_calculator.py --max_sequence_length 8192 --batch_size 1024 --dtype int32 --num_tokens 1000000000000
```

and you will see that you will require 119,210 batches. and the total size of the batches will be around 3725GB

## Batch Memory Calculator
This is a pretty useful utility where you can provide the following information

- batch size
- data type
- max sequence length
- number of batches

So for example, to calculate the number of batches (and size of disk) for 1 trillion tokens in a batch of 1024, with a max sequence length of 8192.  you can run the following command


```bash
python tools/batch/batch_memory_calculator.py --max_sequence_length 8192 --batch_size 1024 --dtype int32 --num_batches 119210 
```

and you will see that you will require 119,210 batches. and the total size of the batches will be around 3725GB, and the memory consumption will be 32MB per batch.
