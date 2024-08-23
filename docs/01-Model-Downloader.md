# Model Downloader
This is a simple utility that allows you to download a model from huggingface hub, and stick it the huggingface cache.  This is pretty useful if you just want to kick off a download and use it later on with another tool.

```bash
python tools/models/model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/models/model_downloader.py --model ibm-granite/granite-3b-code-instruct
```