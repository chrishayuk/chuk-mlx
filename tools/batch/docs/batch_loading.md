# Batch Loading
In order to be able to test that batch loading works.
I've created a simple utility that will load the random batch that we loaded earlier

```bash
python tools/batch/batch_loader_test.py --output_directory sample_data/batches/chuk_random/
```

From this test we can see that we can cope with loading around 625 million tokens per second, or 75 batches per second or 8.3 million tokens in 0.01 seconds.  in this case it's loading about 4.6GB a second

Batch loading is not an issue or a bottleneck