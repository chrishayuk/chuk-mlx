# Model Tools
This is a collection of tools that allow you to:

- download a model
- visualize the configuration of a model
- visualize the layers of a model

## Model Downloader
This is a simple utility that allows you to download a model from huggingface hub, and stick it the huggingface cache.  This is pretty useful if you just want to kick off a download and use it later on with another tool.

```bash
python tools/models/model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.2
```

or

```bash
python tools/models/model_downloader.py --model ibm-granite/granite-3b-code-instruct
```

## Model Viewer - Show Layers
This is a simple utility that allows you to print the configuration of a model

```bash
python tools/models/model_viewer.py --model ibm-granite/granite-3b-code-instruct --show-config
```

## Model Viewer - Show Config
This is a simple utility that allows you to print the configuration of a model

```bash
python tools/models/model_viewer.py --model ibm-granite/granite-3b-code-instruct --show-layers
```

## Print Layer Modules
This is a simple utility that allows you to print the modules of a layer.

```bash
python tools/models/print_layer_modules.py --model ibm-granite/granite-3b-code-instruct
```

## Model Visualizer
This is a simple utility that allows you to visualize the layers of a model graphically.

```bash
python tools/models/model_visualizer.py --model ibm-granite/granite-3b-code-instruct
```