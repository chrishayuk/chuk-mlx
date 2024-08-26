# Viewing Batches
I've created 2 tools, to help you view the contents of a batch file

- Batch Viewer
- Batch Viewer CLI

## Batch Viewer
The batch viewer will provide a pretty grid showing the tokens for a partiular batch.  It will not show the full batch, but rather just upto the last token.

If you wish to see the full contents of the batch, then the batch viewer cli is probably more appropriate

```bash
python tools/batch/batch_viewer.py --batch_file ./output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/batch_viewer.py --batch_file ./sample_data/batches/lazyfox/lazyfox_batch_0001.npz --tokenizer lazyfox
```

or

```bash
python tools/batch/batch_viewer.py --batch_file ./sample_data/batches/math/math_batch_0001.npz --tokenizer math
```

## Batch Viewer CLI
The batch viewer cli shows every token and it's decoded version in the batch.

```bash
python tools/batch/batch_viewer_cli.py --batch_file ./output/batches/calvin/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/batch_viewer_cli.py --batch_file ./sample_data/batches/lazyfox/lazyfox_batch_0001.npz --tokenizer lazyfox
```

or

```bash
python tools/batch/batch_viewer_cli.py --batch_file ./sample_data/batches/math/math_batch_0001.npz --tokenizer math
```
