# Batch Diagnostics

## Random Batch Generation
In order to be able to test that batch generation works.
I've created a simple utility that will generate a random batch with junk tokens.

If you wish to run this, you can

```bash
python batch_generator_random.py
```

In this particular case, it will generate a single batch, with 1024 sequences (rows), with a max sequence length of 8,192 tokens.  This results in a batch of 8,388,608 tokens, i.e 8.3 Million tokens.

It takes about 0.06 seconds to generate this batch

And the batch is about 67MB in size.

If you wish to analyze the batch (it's junk). you can run

```bash
python batch_analyzer.py --batch_file ./output/chuk_random_batch_0001.npz --tokenizer ibm-granite/granite-3b-code-instruct
```

If you wish to view the batch (it's junk), you can run

```bash
python batch_viewer.py --batch_file ./output/chuk_random_batch_0001.npz --tokenizer ibm-granite/granite-3b-code-instruct
```

or via the cli

```bash
python batch_viewer_cli.py --batch_file ./output/chuk_random_batch_0001.npz --tokenizer ibm-granite/granite-3b-code-instruct
```

## Batch Loading
In order to be able to test that batch loading works.
I've created a simple utility that will load the random batch that we loaded earlier

In this case it will load the random batch that we generated...

```bash
python batch_loader_test.py 
```

From this test we can see that we can cope with loading around 625 million tokens per second, or 75 batches per second or 8.3 million tokens in 0.01 seconds.  in this case it's loading about 4.6GB a second

Batch loading is not an issue or a bottlenexl