# Model Downloader
This is a simple utility that allows you to download a model from huggingface hug, and stick it the huggingface cache.
This is pretty useful if you just want to kick off a download and use it later on with another tool


```bash
python model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.2
```

### Code Models
The following shows how to download the IBM code models

```bash
python model_downloader.py --model ibm-granite/granite-3b-code-instruct
```