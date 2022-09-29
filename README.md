# Barlow Graph Auto Encoder

### Setup

Easiest way is to use docker and vscode.

1. Build the image from given file as:

```
docker build -t bgae:latest -f Dockerfile .
```

2. Use vscode remote-container to open the project inside docker container.
Afterward, you can execute `run_for_dataset.py` for a single dataset, or `run_hptune.py` if you want to play around with hyperparameter tuning.

### Config files
`dataset_config.yaml` is used for `run_for_dataset.py` and `hptune_config.yaml` is used for `run_hptune.py`.
In both the files, there is a base configuration that can be patched by dataset-specific configurations as demonstrated by the examples inside these config files.