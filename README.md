# FedELMY: One-Shot Sequential Federated Learning for Non-IID Data by Enhancing Local Model Diversity

This repository is the official implementation of the ACM Multimedia 2024 paper "FedELMY: One-Shot Sequential Federated Learning for Non-IID Data by Enhancing Local Model Diversity".

## Supplementary Material

The supplementary material for the paper can be found [here](FedELMY_SUPP.pdf).

## Code

### Requirements
- This codebase is written for python3 (used Python 3.7.4 while implementing).

With Anaconda, you can create an environment `FedELMY` with the following command:

```bash
conda create -n FedELMY python=3.7.4
```

then, activate the environment with:

```bash
conda activate FedELMY
```

- We use Pytorch version of 1.13.1, 11.6 CUDA version.
- To install necessary Python packages,

```bash
pip install -r requirements.txt
```

### How to Run Codes?

The sample code is as follows:

```shell
python3 FedELMY.py --model resnet18 --dataset cifar10 --num_users 10 --betas 0.5 --num_models 5 --alpha 1 --beta 1 --seed 1  --device cuda:0
```
Here is an explanation of some parameters,
- `--model`: model architecture to be used (e.g., `cnn` or `resnet18`).
- `--dataset`: name of the datasets (e.g., `cifar10`, `pacs`, `oc10` or `tiny`).
- `--num_users`: the number of total clients (e.g., 10, 20, 50).
- `--betas`: concentration parameter beta for latent Dirichlet Allocation when splitting data (default: 0.5).
- `--num_models`: the number of models in the model pool (e.g., 5, 10, 20).
- `--alpha`: the scale control hyperparameter of the distance between the current model to all models within the model pool. (default: `1`).
- `--beta`: the scale control hyperparameter of the distance between the current model to the first model of the model pool. (default: `1`).
- `--seed`: the random seed to be used (e.g., `1`, `2`, `3`).
- `--device`: the device to be used for training (e.g., `cuda:0` or `cpu`).

**Note that the same random seed must be fixed for fair comparison. Because different random seeds mean that the data distribution on each client is different.**


### Citing this work

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{wangfedelmy,
  title={FedELMY: One-Shot Sequential Federated Learning for Non-IID Data by Enhancing Local Model Diversity},
  author={Naibo Wang, Yuchen Deng, Wenjie Feng, Shichen Fan, Jianwei Yin, and See-Kiong Ng},
  booktitle={ACM Multimedia},
  year={2024},
  doi={10.1145/3664647.3681054}
}
```

### Project Structure

- `/checkpoints`

    Used to store model checkpoints during training, allowing for recovery and resuming training.

- `/dataset_info`

    Contains metadata and information regarding datasets used by the models.

- `/helpers`
  - `__init__.py`: Indicates that this directory is a Python package and can contain common initializations for the helper's module.
  - `comm_helpers.py`: Provides helper functions for common tasks across the project.
  - `datasets.py`: Contains functions related to loading and processing datasets.
  - `office_caltech_10.py`: Specific functions for handling the Office-Caltech 10 dataset.
  - `optimizers.py`: Custom or modified optimization algorithms used in training models.
  - `pacs.py`: Functions tailored for the PACS dataset, often used in domain adaptation.
  - `sam.py`: Implements the Sharpness-Aware Minimization (SAM) optimization technique.
  - `utils.py`: General utility functions that are used across different modules.

- `/models`
  - `__init__.py`: Initializes the model's directory as a Python package.
  - `nets.py`: Contains definitions for different neural network architectures.
  - `resnet.py`: Contains the implementation of Residual Network architectures.
  - `vit.py`: Implementation of Vision Transformer models.
  - `wide_resnet.py`: Wide Residual Network implementations, a variant of ResNet with more width.
  - `wrn.py`: Another variant for Wide Residual Networks.

- `/results`
  
    Directory for storing the output results from model evaluations, such as accuracy metrics and loss values.

- `/`
  - `hps.py`: Defines hyperparameters for hyperparameter selection.
  - `FedELMY.py`: The main script with the core logic for the FedELMY framework's functionality.
  - `loop_df_fl.py`: Basic functions for running the FedELMY framework.
  - `README.md`: Provides documentation and an overview of the project.
  - `requirements.txt`: Specifies all the necessary Python packages for the project.
  - `warmup_config.py`: Configuration settings for the warmup phase at the beginning of training.
