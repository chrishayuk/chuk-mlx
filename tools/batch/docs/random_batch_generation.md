# Random Batch Generation
In order to be able to test that batch generation works.
I've created a simple utility that will generate a random batch with junk tokens.

If you wish to run this, you can

```bash
python tools/batch/batch_generator_random.py
```

This will creatw a file in the following location.

```
./output/batches/chuk_random
```

In this particular case, it will generate a single batch, with 1024 sequences (rows), with a max sequence length of 8,192 tokens.  This results in a batch of 8,388,608 tokens, i.e 8.3 Million tokens.

It takes about 0.06 seconds to generate this batch

And the batch is about 67MB in size.

If you wish to analyze the batch (it's junk). you can use the batch_analyzer or batch viewer tools.