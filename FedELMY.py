#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import datetime
import json
import os
import random
import shutil
import sys
import time
import warnings
from hps import *

import math
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb

from helpers.datasets import partition_data
from helpers.utils import get_dataset, mean_average_weights, DatasetSplit, KLDiv, setup_seed, test, \
    federated_average_weights
from loop_df_fl import get_model, LocalUpdate, Ensemble
from models.nets import CNNCifar, CNNMnist, CNNCifar100, CNNPACS
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data.dataset import random_split

from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
from warmup_config import warmup_config
# import wandb
from commandline_config import Config


warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)

preset_config = {
    "warmup_epochs": -1,# When to start split learning by different hyperparameters
    "fedavgEpochs": 1,# number of rounds of training
    "num_users": 10,# number of users: K
    "num_classes": 10,# number of classes
    "num_models": 5,# number of models per user for model pool
    "frac": 1,# the fraction of clients: C
    "local_ep": -1,# the number of local epochs: E
    "max_hp_count": 9999,# the number of local epochs: E
    "local_bs": 128,# local batch size: B
    "lr": 0.01,# learning rate
    "image_size": -1,
    "validation_ratio": 0.1,# Validation dataset ratio
    "momentum": 0.9,# SGD momentum (default: 0.5)
    "weight_decay": 1e-4,# SGD weight decay (default: 1e-4)
    "optimizer": "-1",
    "record_distances": 0,
    "note": "",

    "dataset": "cifar10",# name of dataset
    "random_position": "inside",# Position of random
    "iid": 0,# Default set to IID. Set to 0 for non-IID.
    "mu": 1,# mu for fedprox
    "optimization_method": "none",
    "alpha": 1,# alpha for the regularization term
    "beta": 1,# beta for the regularization term
    "order": 1,# order of domain shift tasks
    "save_every_model": 0,#

    "adv": 1,# scaling factor for adv loss
    "bn": 1,# scaling factor for BN regularization
    "oh": 1,# scaling factor for one hot loss (cross entropy)
    "act": 0,# scaling factor for activation loss used in DAFL
    "save_dir": "run/synthesis",
    "partition": "dirichlet",#
    "betas": 0.3,# Split distribution, If betas is set to a smaller value, then the partition is more unbalanced

    "lr_g": 1e-3,#initial learning rate for generation
    "T": 20,#
    "g_steps": 30,# number of iterations for generation
    "batch_size": 256,# number of total iterations in each epoch
    "nz": 256,# number of total iterations in each epoch
    "synthesis_batch_size": 256,

    "seed": 1,# seed for initializing training
    "epochs": 50,
    "type": "pretrain",
    "model": "cnn",
    "other": "",
    "device": "cuda:0",# Device ID
    "id": "0"# File ID
}

if __name__ == '__main__':
    config = Config(preset_config, name='Federated Learning Experiments')
    print(config)
    if not torch.cuda.is_available():
        config.device = "cpu"
        print("CUDA is not available, use CPU.")
    setup_seed(config.seed)
    # pdb.set_trace()
    # BUILD MODEL
    start_time = time.time()
    global_model = get_model(config)
    init_weights = copy.deepcopy(global_model.state_dict())
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    global_model.train()
    fedavg_accs = []
    client_accs = []
    if config.id == "0":
        id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    else:
        id = config.id
    print("id: {}".format(id))
    time.sleep(3)
    if config.fedavgEpochs == 1:
        hps = hyperparameters_one_shot[config.dataset]
    else:
        hps = hyperparameters[config.dataset]
    fedavg_model_weights = []
    saved_model_weights_pool = []
    # ===============================================
    model_weights_pool = []
    for i in range(config.fedavgEpochs):  # FEDAVG TEST
        local_weights = []
        user_avg_weights = []
        users = []
        saved_datasets = []
        acc_list = []
        max_accs = []
        best_model_weights = []
        val_accs = []
        client_losses = []
        if i == 0:
            # Client 0 warmup
            if config.warmup_epochs != -1:
                warmup_epochs = config.warmup_epochs
            else:
                warmup_epochs = warmup_config[config.dataset][config.model][0]
            hyperparameter = hps[0]
            train_dataset, val_dataset, test_dataset, user_groups, val_user_groups, training_data_cls_counts = partition_data(
                config.dataset, config.partition, beta=config.betas, num_users=config.num_users,
                transform=hyperparameter["transform"], order=config.order)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                      shuffle=False, num_workers=4)
            local_model = LocalUpdate(args=config, dataset=train_dataset, val_dataset=val_dataset,
                                      idxs=user_groups[0], val_idxs=val_user_groups[0], test_loader=test_loader)
            training_set, valid_set = local_model.get_datasets()
            global_model.load_state_dict(init_weights)
            print("Start Warm Up")
            warmup_weights, local_acc_list, best_epoch, max_val_acc, local_loss_list = local_model.update_weights(
            copy.deepcopy(global_model), config.device, hyperparameter, local_ep=warmup_epochs, optimize=False, args=config)
            t_warmup_weights = copy.deepcopy(warmup_weights)
            model_weights_pool.append(t_warmup_weights)

        for idx in range(config.num_users):
            train_dataset, val_dataset, test_dataset, user_groups, val_user_groups, training_data_cls_counts = partition_data(
                config.dataset, config.partition, beta=config.betas, num_users=config.num_users,
                transform=hyperparameter["transform"], order=config.order)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                      shuffle=False, num_workers=4)
            for m in range(config.num_models):
                print("Now training model {} for user {}".format(m, idx))
                local_model = LocalUpdate(args=config, dataset=train_dataset, val_dataset=val_dataset,
                                          idxs=user_groups[idx], val_idxs=val_user_groups[idx], test_loader=test_loader)
                model_weights_pool, local_acc_list, best_epoch, max_val_acc, local_loss_list = local_model.update_weights_model_pool(
                    copy.deepcopy(global_model), config.device, hyperparameter, model_weights_pool, random_position=config.random_position, args=config)
            saved_model_weights_pool.extend(model_weights_pool)
            model_weights_pool = [mean_average_weights(model_weights_pool)]

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(saved_model_weights_pool,
               'checkpoints/{}_{}clients_{}_{}_{}_{}.pkl'.format(config.dataset, config.num_users, config.betas,
                                                                 config.partition, config.model, id))

    global_weights = mean_average_weights(model_weights_pool)
    global_model.load_state_dict(global_weights)
    print("One-Shot MeanAvg Accuracy:")
    meanavg_test_acc, meanavg_test_loss = test(global_model, test_loader, config.device)

    model_list = []
    for i in range(len(model_weights_pool)):
        net = copy.deepcopy(global_model)
        net.load_state_dict(model_weights_pool[i])
        model_list.append(net)
    ensemble_model = Ensemble(model_list)
    print("Ensemble Accuracy:")
    ensemble_test_acc, ensemble_test_loss = test(ensemble_model, test_loader, config.device)
    max_acc = []
    for sub in acc_list:
        max_acc.append(max(sub))
    last_acc = []
    for sub in acc_list:
        last_acc.append(sub[-1])
    output = {
        "id": id,
        "seed": config.seed,
        "fedavgEpochs": config.fedavgEpochs,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": config.dataset,
        "model": config.model,
        "num_users": config.num_users,
        "betas": config.betas,
        "partition": config.partition,
        "warmup_config": warmup_config[config.dataset],
        "file": os.path.basename(__file__),
        "meanavg_test_acc": meanavg_test_acc,
        "meanavg_test_loss": meanavg_test_loss,
        "ensemble_test_acc": ensemble_test_acc,
        "ensemble_test_loss": ensemble_test_loss,
        "hyperparameters": hps,
        "data_cls_counts": str(training_data_cls_counts),
        "args": str(config),
        "local_bs": config.local_bs,
        "validation_ratio": config.validation_ratio,
        "note": config.note,
        "order": config.order,
        "client_losses": client_losses,
        "val_accs": val_accs,
        "alpha": config.alpha,
        "beta": config.beta,
    }
    if not os.path.exists('results'):
        os.makedirs('results')
    json.dump(output, open(
        'results/{}_{}clients_{}_{}_{}.json'.format(config.dataset, config.num_users, config.betas, config.model, id), 'w'))
    output = json.dumps(output, indent=4)
    print(output)
    print("One-Shot MeanAvg Accuracy:")
    meanavg_test_acc, meanavg_test_loss = test(global_model, test_loader, config.device)
    end_time = time.time()
    print("Total Time Cost: {:.2f}s".format(end_time - start_time))
    # ===============================================

