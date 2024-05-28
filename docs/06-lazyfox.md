# Introduction
Lazyfox is a very simple language model that is used to train the model to predict the next token of the classic phrase "the quick brown fox jumps over the lazy dog".

## lazyfox dataset
the lazyfox dataset in jsonl format can be found at 'sample_data/lazyfox_train.jsonl'

## lazyfox tokenizer
there is a very simple tokenizer for lazyfox which just consists of the words of the phrase, the tokenizer can be found at 'chuk_tokenizers/lazyfox_tokenizer.py'.  the model_hub loader natively supports local tokenizers in this folder as well as huggingface tokenizers.

### printing tokens for a prompt
if you wish to test the lazyfox tokenizer you can use the print_tokens utility

```bash
python print_tokens.py --tokenizer lazyfox_tokenizer --prompt "the quick brown fox jumps over the lazy dog"
```


## Batch Generation
Since the lazyfox model is a blank model, it uses pre-trained batchfiles to train the model.
This section shows how to generate batches for training the lazyfox model.

### Generating the input batchfiles
To generate the batchfiles for the inputs then run the following.

```bash
python batch_generator_pretrain.py --input_files ./sample_data/lazyfox_train.jsonl --tokenizer lazyfox_tokenizer --output_directory ./output/lazyfox/batches --file_prefix lazyfox --max_sequence_length 16 --batch_size 1
```

### Generating the target batchfiles
To generate the batchfiles for the targets then run the following.

```bash
python batch_pretrain_target_shift_generator.py --input_directory ./output/lazyfox/batches --batch_prefix lazyfox
```

### Diagnosing the Batchfiles
If you wish to check the lazyfox batchfiles have been generated correctly you can run the following.

#### Batch Loader
This will test that the generated batches load properly, also giving the diagnostic information of the batches including load time, memory usage, number of tokens.

```bash
python batch_loader_test.py --output_directory ./output/lazyfox/batches --file_prefix lazyfox
```

#### Batch Analyzer
The following allows you to perform a more detailed analysis of the generated batches including analysis of the number of tokens per row, percentage of tokens that are padding tokens

to run this against the generated input batch, you can run:

```bash
python batch_analyzer.py --batch_file output/lazyfox/batches/lazyfox_batch_0001.npz --tokenizer lazyfox_tokenizer
```

#### Batch Viewer
The following allows you to view the generated batches.

to run this against the generated input batch, you can run:

```bash
python batch_viewer.py --batch_file output/lazyfox/batches/lazyfox_batch_0001.npz --tokenizer lazyfox_tokenizer
```

# Execute Lazy Fox
You can perform a pretrain of lazyfox using the lazyfox script

```bash
python lazyfox.py
```

or you can use the generic pretrain script

```bash
python pretrain.py --config ./training_config/pretrain/lazyfox.yaml
```

# infer lazy fox (pre-train)

```bash
python infer.py --model lazyfox --checkpoint ./output/lazyfox/checkpoints/checkpoint_epoch_50.npz --prompt "the quick brown"
```