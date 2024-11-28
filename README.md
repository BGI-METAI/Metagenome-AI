# Metagenome-AI

## Overview

The framework for labeling and classification of protein sequences based on Large Protein models.

## Configuration Parameters

| Parameter Name        | Description                                                                                                                                                                                                | Required | Default Value    |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|------------------|
| `model_type`          | Type of embedding model to be used. Options: `"ESM"`, `"ESM3"`, `"PTRANS"`, `"PVEC"`.                                                                                                                      | Yes      | `None`           |
| `classifier_type`     | Type of classification head to be used. Options: `"MLP"`, `"XGBoost"`.                                                                                                                                     | Yes      | `None`           |
| `program_mode`        | Mode in which the program will run. Options: `"ONLY_STORE_EMBEDDINGS"` (only store embeddings), `"TRAIN_PREDICT_FROM_STORED"` (train classifier from stored embeddings), or `"RUN_ALL"` (store and train). | Yes      | `"RUN_ALL"`      |
| `train`               | Path to the training dataset.                                                                                                                                                                              | Yes      | `None`           |
| `valid`               | Path to the validation dataset. If not available, the test dataset path can be reused.                                                                                                                     | Yes      | `None`           |
| `test`                | Path to the test dataset.                                                                                                                                                                                  | Yes      | `None`           |
| `emb_dir`             | Directory to store embeddings.                                                                                                                                                                             | Yes      | `None`           |
| `model_folder`        | Directory to save model checkpoints.                                                                                                                                                                       | Yes      | `None`           |
| `model_basename`      | Name under which to save model checkpoints.                                                                                                                                                                | Yes      | `None`           |
| `wandb_key`           | API key for Weights and Biases tracking.                                                                                                                                                                   | Yes      | `None`           |
| `batch_size`          | Batch size for classifier training.                                                                                                                                                                        | No       | `32`             |
| `num_epochs`          | Number of epochs for MLP classifier training.                                                                                                                                                              | No       | `10`             |
| `lr`                  | Learning rate for the MLP classifier optimizer.                                                                                                                                                            | No       | `0.001`          |
| `hidden_layers`       | List specifying hidden layer sizes for the MLP classifier. Example: `[1024, 512]`.                                                                                                                         | No       | `None`           |
| `early_stop_patience` | Number of epochs without improvement after which MLP training stops.                                                                                                                                       | No       | `4`              |
| `objective`           | Objective function for XGBoost classifier training.                                                                                                                                                        | No       | `multi:softmax`  |
| `n_estimators`        | Number of trees for XGBoost classifier training.                                                                                                                                                           | No       | `10`             |
| `eta`                 | Learning rate for XGBoost classifier.                                                                                                                                                                      | No       | `0.001`          |
| `early_stop`          | Number of rounds without improvement after which XGBoost training stops.                                                                                                                                   | No       | `4`              |
| `max_depth`           | Maximum tree depth for XGBoost classifier.                                                                                                                                                                 | No       | `8`              |
| `eval_metric`         | Evaluation metric for XGBoost classifier training.                                                                                                                                                         | No       | `mlogloss`       |
| `verbosity`           | Verbosity level for XGBoost classifier training. Higher values produce more detailed output.                                                                                                               | No       | `1`              |
| `max_tokens`          | Maximum number of tokens per dataset split for embedding generation.                                                                                                                                       | No       | `2500`           |
| `classifier_path`     | Path to a pre-trained classifier checkpoint. If provided, training will be skipped.                                                                                                                        | No       | `None`           |
| `log_dir`             | Directory for saving log files.                                                                                                                                                                            | No       | `./logs/`        |
| `pred_dir`            | Directory for saving predictions of the final model.                                                                                                                                                       | No       | `./predictions/` |
| `umap_num_samples`    | The number of random samples to select for UMAP visualization. If not provided, UMAP visualization will be skipped.                                                                                        | No       | `None`           |

### Notes:
- Required parameters **must** be present in the config file for the program to run.
- If a parameter is not set in the config file, the default value will be used, if available.

## Installation
After cloning the repository:
```cd Metagenome-AI
conda create -n mai python==3.9
conda activate mai
pip install -r requirements.txt
```
## Usage

`python src/train.py -c src/configs/config_sample.json`


## License

MIT

## Contact

For all suggestions please create the GitHub issue or contact vladimirkovacevic@genomics.cn.
