import os
import logging
import pickle
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import random

from helpers.office_caltech_10 import office_caltech_10
from helpers.pacs import pacs


def load_data(dataset, transform=None):
    data_dir = './dataset'
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]))
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    elif dataset == "cifar10":
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform)
        val_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                        transform=transforms.Compose(
                                            [
                                                # transforms.RandomHorizontalFlip(), # 随机会使得每次测试验证集的准确率不一样
                                                transforms.ToTensor(),
                                            ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))

    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        val_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
    else:
        raise NotImplementedError
    if dataset == "svhn":
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels
    elif dataset == "tiny":
        X_train, y_train = train_dataset, train_dataset.targets
        X_test, y_test = test_dataset, test_dataset.targets
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    if "cifar10" in dataset or dataset == "svhn":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        # X_train = X_train.data.numpy()
        y_train = np.array(y_train)
        # X_test = X_test.data.numpy()
        y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset, val_dataset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.5, num_users=10, transform=None, validation_ratio = 0.1, order=1):
    n_parties = num_users
    if dataset == "pacs":
        photo_dataset, art_dataset, cartoon_dataset, sketch_dataset, val_photo_dataset, val_art_dataset, val_cartoon_dataset, val_sketch_dataset, test_dataset = pacs(transform)
        # if order == 1:
        #     train_dataset = [photo_dataset, art_dataset, cartoon_dataset, sketch_dataset]
        #     val_dataset = [val_photo_dataset, val_art_dataset, val_cartoon_dataset, val_sketch_dataset]
        # elif order == 2:
        #     train_dataset = [art_dataset, cartoon_dataset, sketch_dataset, photo_dataset]
        #     val_dataset = [val_art_dataset, val_cartoon_dataset, val_sketch_dataset, val_photo_dataset]
        # elif order == 3:
        #     train_dataset = [cartoon_dataset, sketch_dataset, photo_dataset, art_dataset]
        #     val_dataset = [val_cartoon_dataset, val_sketch_dataset, val_photo_dataset, val_art_dataset]
        # elif order == 4:
        #     train_dataset = [sketch_dataset, photo_dataset, art_dataset, cartoon_dataset]
        #     val_dataset = [val_sketch_dataset, val_photo_dataset, val_art_dataset, val_cartoon_dataset]
        # elif order == 5:
        #     train_dataset = [photo_dataset, art_dataset, sketch_dataset, cartoon_dataset]
        #     val_dataset = [val_photo_dataset, val_art_dataset, val_sketch_dataset, val_cartoon_dataset]
        # elif order == 6:
        #     train_dataset = [art_dataset, sketch_dataset, cartoon_dataset, photo_dataset]
        #     val_dataset = [val_art_dataset, val_sketch_dataset, val_cartoon_dataset, val_photo_dataset]
        # elif order == 7:
        #     train_dataset = [sketch_dataset, cartoon_dataset, photo_dataset, art_dataset]
        #     val_dataset = [val_sketch_dataset, val_cartoon_dataset, val_photo_dataset, val_art_dataset]
        # elif order == 8:
        #     train_dataset = [cartoon_dataset, photo_dataset, art_dataset, sketch_dataset]
        #     val_dataset = [val_cartoon_dataset, val_photo_dataset, val_art_dataset, val_sketch_dataset]
        from itertools import permutations

        datasets = [photo_dataset, art_dataset, cartoon_dataset, sketch_dataset]
        datasets_val = [val_photo_dataset, val_art_dataset, val_cartoon_dataset, val_sketch_dataset]
        datasets_text = ['photo', 'art', 'cartoon', 'sketch']

        train_dataset = []
        val_dataset = []

        order_permutations = list(permutations(datasets, len(datasets)))
        order_permutations_val = list(permutations(datasets_val, len(datasets_val)))
        order_permutations_text = list(permutations(datasets_text, len(datasets_text)))

        if order > 0 and order <= len(order_permutations):
            train_dataset = list(order_permutations[order - 1])
            val_dataset = list(order_permutations_val[order - 1])
            print("Order: ", order)
        else:
            print("Error - order out of range")

        train_data_cls_counts = {
            0: {0: len(train_dataset[0])},
            1: {0: len(train_dataset[1])},
            2: {0: len(train_dataset[2])},
            3: {0: len(train_dataset[3])}
        }
        net_dataidx_map = [0, 1, 2, 3]
        val_dataidx_map = [0, 1, 2, 3]
    elif dataset == "oc10": # office-caltech-10
        amazon_dataset, caltech_dataset, dslr_dataset, webcam_dataset, val_amazon_dataset, val_caltech_dataset, val_dslr_dataset, val_webcam_dataset, test_dataset = office_caltech_10(transform)
        train_data_cls_counts = {
            0: {0: len(amazon_dataset)},
            1: {0: len(caltech_dataset)},
            2: {0: len(dslr_dataset)},
            3: {0: len(webcam_dataset)}
        }
        train_dataset = [amazon_dataset, caltech_dataset, dslr_dataset, webcam_dataset]
        val_dataset = [val_amazon_dataset, val_caltech_dataset, val_dslr_dataset, val_webcam_dataset]
        net_dataidx_map = [0, 1, 2, 3]
        val_dataidx_map = [0, 1, 2, 3]
    else:
        X_train, y_train, X_test, y_test, train_dataset, test_dataset, val_dataset = load_data(dataset, transform)
        file_name = "dataset_info/" + dataset + "_" + partition + "_" + str(beta) + "_" + str(num_users) + ".pkl"
        # print(file_name)
        if not os.path.exists(file_name):
            data_size = y_train.shape[0]
            if partition == "iid":
                idxs = np.random.permutation(data_size)
                batch_idxs = np.array_split(idxs, n_parties)
                net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
            elif partition == "dirichlet":
                min_size = 0
                min_require_size = 10
                label = np.unique(y_test).shape[0]
                net_dataidx_map = {}

                while min_size < min_require_size:
                    idx_batch = [[] for _ in range(n_parties)]
                    for k in range(label):
                        idx_k = np.where(y_train == k)[0]
                        np.random.shuffle(idx_k)  # shuffle the label
                        # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                        proportions = np.array(  # 0 or x
                            [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                        min_size = min([len(idx_j) for idx_j in idx_batch])
                for j in range(n_parties):
                    np.random.shuffle(idx_batch[j])
                    net_dataidx_map[j] = idx_batch[j]

            val_dataidx_map = {}
            for i in range(n_parties):
                val_size = int(len(net_dataidx_map[i]) * validation_ratio)
                val_dataidx_map[i] = net_dataidx_map[i][:val_size]
                net_dataidx_map[i] = net_dataidx_map[i][val_size:]
            info = {
                "dataset": dataset,
                "partition": partition,
                "beta": beta,
                "num_users": num_users,
                "net_dataidx_map": net_dataidx_map,
                "val_dataidx_map": val_dataidx_map,
            }
            with open(file_name, 'wb') as f:
                pickle.dump(info, f)
                # print("Save dataset info to " + file_name)
        else:
            with open(file_name, 'rb') as f:
                dataset_info = pickle.load(f)
                assert dataset_info["partition"] == partition
                assert dataset_info["beta"] == beta
                assert dataset_info["num_users"] == num_users
                assert dataset_info["dataset"] == dataset
            net_dataidx_map = dataset_info["net_dataidx_map"]
            val_dataidx_map = dataset_info["val_dataidx_map"]
            print("Use dataset info from " + file_name)
            time.sleep(1)
        train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, val_dataset, test_dataset, net_dataidx_map, val_dataidx_map, train_data_cls_counts
