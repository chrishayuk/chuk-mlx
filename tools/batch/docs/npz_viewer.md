# Viewing NPZ files
I've created 2 tools, to help you view the contents of an npz file

- NPZ Viewer
- NPZ Tokenized viewer

## NPZ Viewer
The npz viewer will allow you to check the following for an npz file.

- whether it's valid
- whether it's loadable with numpy and mlx
- the tensors in the file
- the shape, size and data types of the tensors

```bash
python tools/batch/npz_viewer.py --batch_file ./sample_data/batches/calvin_scale/calvin_batch_0001.npz
```

or

```bash
python tools/batch/npz_viewer.py --batch_file ./sample_data/batches/lazyfox/lazyfox_batch_0001.npz
```

or

```bash
python tools/batch/npz_viewer.py --batch_file ./sample_data/batches/math/math_batch_0001.npz
```

## NPZ Tokenized Viewer
The npz tokenized viewer shows the tensors in the npz file along with their decoded values

```bash
python tools/batch/npz_tokenized_viewer.py --batch_file ./sample_data/batches/calvin_scale/calvin_batch_0001.npz --tokenizer mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/batch/npz_tokenized_viewer.py --batch_file ./sample_data/batches/lazyfox/lazyfox_batch_0001.npz --tokenizer lazyfox
```

or

```bash
python tools/batch/npz_tokenized_viewer.py --batch_file ./sample_data/batches/math/math_batch_0001.npz --tokenizer math
```
