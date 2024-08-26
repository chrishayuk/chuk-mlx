# Batch Analyzer
If you wish to analyze the output of a particular batch you can use the following command:

```bash
python tools/batch/batch_analyzer.py --batch_file ./output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/batch_analyzer.py --batch_file ./sample_data/batches/lazyfox/lazyfox_batch_0001.npz --tokenizer lazyfox
```

or

```bash
python tools/batch/batch_analyzer.py --batch_file ./sample_data/batches/math/math_batch_0001.npz --tokenizer math
```

This will allow you to see the numbr of rows, the number of padding tokens etc for a particular batch