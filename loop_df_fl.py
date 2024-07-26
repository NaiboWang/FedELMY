#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import datetime
import json
import os
import random
import shutil
import sys
import time
import warnings

from helpers.sam import SAM, disable_running_stats, enable_running_stats
from hps import hyperparameters, hyperparameters_one_shot

import math
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb

from helpers.datasets import partition_data
from helpers.utils import get_dataset, mean_average_weights, DatasetSplit, KLDiv, setup_seed, test, \
    federated_average_weights
from models.nets import CNNCifar, CNNMnist, CNNCifar100, CNNPACS, SimpleCNN, SimpleCNNTiny
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data.dataset import random_split

from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224

# import wandb

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)


def scale_to_order_of_magnitude(a, b, scale=1):
    if b == 0:
        return 1
    order_of_magnitude_a = math.floor(math.log10(a))
    order_of_magnitude_b = math.floor(math.log10(b))

    if order_of_magnitude_b > order_of_magnitude_a - scale:
        return 10 ** (order_of_magnitude_b - order_of_magnitude_a + scale)

    return 1


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, val_idxs, test_loader, val_dataset):
        self.args = args
        if args.dataset != "pacs" and args.dataset != "oc10":
            self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                           batch_size=self.args.local_bs, shuffle=True, num_workers=4)
            self.valid_loader = DataLoader(DatasetSplit(val_dataset, val_idxs), batch_size=self.args.local_bs,
                                           shuffle=False)
            self.train_dataset = DatasetSplit(dataset, idxs)
            self.valid_dataset = DatasetSplit(val_dataset, val_idxs)
            print("Length of idxs: {}".format(len(idxs)))
            print("IDXS:", idxs[:30])
        else:
            if type(idxs) == int:
                self.train_loader = DataLoader(dataset[idxs], batch_size=self.args.local_bs, shuffle=True,
                                               num_workers=4)
                print("Use all data for training")
                self.train_dataset = dataset[idxs]
            else:
                self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                               batch_size=self.args.local_bs, shuffle=True, num_workers=4)
                self.train_dataset = DatasetSplit(dataset, idxs)
                print("Use {} data for training".format(len(idxs)))
                print("IDXS:", idxs[:30])
            self.valid_loader = DataLoader(val_dataset[val_idxs], batch_size=self.args.local_bs, shuffle=False,
                                           num_workers=4)
            self.valid_dataset = val_dataset[val_idxs]
        self.test_loader = test_loader

    def get_datasets(self):
        return self.train_dataset, self.valid_dataset

    def update_weights(self, model, device, hp, local_ep=-1, optimize=True, return_model=False, args=None):
        global_model = copy.deepcopy(model)
        global_weight_collector = list(global_model.to(device).parameters())
        model.train()
        if args.model == "cnn":
            hp["lr"] = 0.001
            print("Use lr = 0.001")
        if hp['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=hp["lr"],
                                        momentum=hp['momentum'])
        elif hp['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp['weight_decay'])
        elif hp['optimizer'] == 'sam':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer,
                            lr=hp["lr"], momentum=0.9, weight_decay=hp['weight_decay'])
            print("Use SAM")
        try:
            optimization_method = hp['optimization_method']
        except:
            optimization_method = "none"
        print("Optimization method: {}".format(optimization_method))
        local_acc_list = []
        local_loss_list = []
        criterion = torch.nn.CrossEntropyLoss().to(device)
        max_valid_acc = 0
        best_epoch = 0
        if local_ep != -1:
            local_ep = local_ep
        elif self.args.local_ep != -1:
            local_ep = self.args.local_ep
        else:
            local_ep = hp['local_ep']
        for iter in tqdm(range(local_ep)):
            model.train()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                if hp["optimizer"] != "sam":
                    optimizer.zero_grad()
                output = model(images)
                proximal_term = 0.0
                original_loss = criterion(output, labels)
                if optimization_method == "fedprox" and optimize:
                    for param_index, param in enumerate(model.parameters()):
                        proximal_term += torch.pow(torch.norm(param - global_weight_collector[param_index]), 2)
                    loss = original_loss + hp['mu'] / 2.0 * proximal_term
                elif optimization_method == "none" or not optimize:
                    loss = original_loss
                if hp["optimizer"] == "sam":
                    enable_running_stats(model)
                    criterion(model(images), labels).backward()
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(model)
                    criterion(model(images), labels).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    loss.backward()
                    optimizer.step()

            print("\n\nOriginal loss: {:.4f}, Proximal term: {:.4f}, Loss: {:.4f}\n\n".format(original_loss.item(), proximal_term, loss.item()))
            with torch.no_grad():
                acc_val, loss_val = test(model, self.valid_loader, device)
                local_loss_list.append(loss_val)
                if acc_val > max_valid_acc:
                    max_valid_acc = acc_val
                    best_model = copy.deepcopy(model.state_dict())
                    best_epoch = iter
                    print("Best model updated at epoch {} with Validation Accuracy: {}".format(iter, acc_val))

        model.load_state_dict(best_model)
        acc, test_loss = test(model, self.test_loader, device)
        local_acc_list.append(acc)
        if return_model:
            model.load_state_dict(best_model)
            best_model = copy.deepcopy(model)
        return best_model, local_acc_list, best_epoch, max_valid_acc, local_loss_list

    def update_weights_model_pool(self, model, device, hp, model_weights_pool, local_ep=-1, init='init',
                                  random_position="outside", args=None):
        if args is None:
            alpha = 1.0
            beta = 1.0
        else:
            alpha = args.alpha
            beta = args.beta
        if args.model == "cnn":
            hp["lr"] = 0.001
            print("Use lr = 0.001")
        print("Alpha: {}, Beta: {}".format(alpha, beta))
        weights = mean_average_weights(model_weights_pool)
        model.load_state_dict(weights)
        model.train()
        model_weights_pool.append(model.state_dict())
        model_pool = []
        model_pool_collectors = []
        for model_weights in model_weights_pool:
            t_model = copy.deepcopy(model)
            t_model.load_state_dict(model_weights)
            model_pool.append(t_model)
            model_pool_collectors.append(list(t_model.to(device).parameters()))
        if init == "init":
            f_init = model_pool_collectors[0]
            print("Use init weights as f_init")
        elif init == "average":
            f_init_weights = mean_average_weights(model_weights_pool[:-1])
            f_init_model = copy.deepcopy(model)
            f_init_model.load_state_dict(f_init_weights)
            f_init = list(f_init_model.to(device).parameters())
            print("Use average weights as f_init")
        if hp['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=hp["lr"],
                                        momentum=hp['momentum'])
        elif hp['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp['weight_decay'])
        elif hp['optimizer'] == 'sam':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer,
                            lr=hp["lr"], momentum=0.9, weight_decay=hp['weight_decay'])
            print("Use SAM")
        local_acc_list = []
        local_loss_list = []
        criterion = torch.nn.CrossEntropyLoss().to(device)
        max_valid_acc = 0
        best_epoch = 0
        if local_ep != -1:
            local_ep = local_ep
        elif self.args.local_ep != -1:
            local_ep = self.args.local_ep
        else:
            local_ep = hp['local_ep']
        for iter in tqdm(range(local_ep)):
            model.train()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                if hp["optimizer"] != "sam":
                    optimizer.zero_grad()
                losses = 0.0
                output = model(images)
                original_loss = criterion(output, labels)
                losses += original_loss
                dist_1 = 0.0
                for i in range(len(model_pool_collectors)):
                    for param_index, param in enumerate(model.parameters()):
                        dist_1 += torch.pow(torch.norm(param - model_pool_collectors[i][param_index]), 2)
                dist_1 = dist_1 / len(model_pool_collectors)  # dist (f_i, M_p)
                dist_2 = 0.0
                for param_index, param in enumerate(model.parameters()):
                    dist_2 += torch.pow(torch.norm(param - f_init[param_index]), 2)  # dist(f_i,f_init)
                scale = 1
                a = scale_to_order_of_magnitude(original_loss.item(), dist_2.item(), scale=scale)
                loss = losses - alpha * dist_1 / a + beta * dist_2 / a  # tiny/oc10

                if hp["optimizer"] == "sam":
                    enable_running_stats(model)
                    criterion(model(images), labels).backward()
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(model)
                    criterion(model(images), labels).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    loss.backward()
                    optimizer.step()
            print("\n\nOriginal loss: {:.4f}, dist_1: {:.4f}, dist_2: {:.4f}, Loss: {:.4f}\n\n".format(
                original_loss.item(),
                dist_1,
                dist_2,
                loss.item()))
            with torch.no_grad():
                acc_val, loss_val = test(model, self.valid_loader, device)
                local_loss_list.append(loss_val)
                if acc_val > max_valid_acc:
                    max_valid_acc = acc_val
                    best_model = copy.deepcopy(model.state_dict())
                    best_epoch = iter
                    print("Best model updated at epoch {} with Validation Accuracy: {}".format(iter, acc_val))

        print("Test Accuracy for whole test loader of last model:")
        acc, test_loss = test(model, self.test_loader, device)
        model.load_state_dict(best_model)
        print("Test Accuracy for whole test loader of best model at epoch {}:".format(best_epoch))
        acc, test_loss = test(model, self.test_loader, device)
        local_acc_list.append(acc)
        model_weights_pool[-1] = copy.deepcopy(best_model)
        avg_model = copy.deepcopy(model)
        avg_model.load_state_dict(mean_average_weights(model_weights_pool))
        print("Test Accuracy for whole test loader of avg model of whole model pool:")
        acc, test_loss = test(avg_model, self.test_loader, device)
        return model_weights_pool, local_acc_list, best_epoch, max_valid_acc, local_loss_list

def get_new_model_weights(model_pool_weights, weights):
    new_model_weights = {}
    for key in model_pool_weights[0].keys():
        new_model_weights[key] = sum(
            weight * model_weights[key] for model_weights, weight in zip(model_pool_weights, weights))
    return new_model_weights


def generate_random_array(n):
    arr = np.random.rand(n)
    arr /= arr.sum()
    return arr


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)

    parser.add_argument('--warmup_epochs', type=int, default=-1,
                        help="When to start split learning by different hyperparameters")
    parser.add_argument('--fedavgEpochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes")
    parser.add_argument('--num_models', type=int, default=5,
                        help="number of models per user for model pool")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=-1,
                        help="the number of local epochs: E")
    parser.add_argument('--max_hp_count', type=int, default=9999,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--image_size', type=int, default=-1,
                        help='image size')
    parser.add_argument('--validation_ratio', type=float, default=0.1, help='Validation dataset ratio')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='-1',
                        help='optimizer')
    parser.add_argument('--record_distances', type=int, default=0,
                        help='record_distances')
    parser.add_argument('--note', type=str, default='',
                        help='note')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--random_position', type=str, default='inside', help="Position of random")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--mu', default=1, type=float, help='mu for fedprox')
    parser.add_argument('--optimization_method', type=str, default='none')
    parser.add_argument('--alpha', default=1, type=float, help='alpha for the regularization term')
    parser.add_argument('--beta', default=1, type=float, help='beta for the regularization term')
    parser.add_argument('--order', default=1, type=int, help='order of domain shift tasks')
    parser.add_argument('--save_every_model', type=int, default=0)

    # Data Free
    parser.add_argument('--adv', default=1, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=1, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=1, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--betas', default=0.3, type=float,
                        help='Split distribution, If betas is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=20, type=float)
    parser.add_argument('--g_steps', default=30, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='epochs')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="cnn", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='Device ID')
    parser.add_argument('--id', default="0", type=str,
                        help='File ID')
    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.dataset == "pacs":
        args.num_classes = 7
        args.num_users = 4
    elif args.dataset == "oc10":
        args.num_classes = 10
        args.num_users = 4
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.num_users = 10
    elif args.dataset == "cifar10":
        args.num_classes = 10
    if args.model == "mnist_cnn":
        global_model = CNNMnist().to(args.device)
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().to(args.device)
    elif args.model == "cnn":
        if args.dataset == "cifar10":
            global_model = CNNCifar(args.num_classes).to(args.device)
            print("Use CNNCifar")
        elif args.dataset == "tiny":
            global_model = SimpleCNNTiny(num_classes=args.num_classes).to(args.device)
            print("Use tinyCNN")
        else:
            global_model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.num_classes).to(
                args.device)
            print("Use SimpleCNN")
    elif args.model == "cnn_pacs":
        global_model = CNNPACS().to(args.device)
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().to(args.device)
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().to(args.device)
    elif args.model == "resnet18":
        global_model = resnet18(num_classes=args.num_classes).to(args.device)

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.to(args.device)
        global_model = torch.nn.DataParallel(global_model)
    return global_model