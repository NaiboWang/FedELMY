import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

NUM_CLASSES = 10
DATASETS_NAMES = ['amazon', 'caltech', 'dslr', 'webcam']
CLASSES_NAMES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

DIR_AMAZON = 'data/office_caltech_10/TRAIN/amazon'
DIR_CALTECH = 'data/office_caltech_10/TRAIN/caltech'
DIR_DSLR = 'data/office_caltech_10/TRAIN/dslr'
DIR_WEBCAM = 'data/office_caltech_10/TRAIN/webcam'
DIR_VAL_AMAZON = 'data/office_caltech_10/VAL/amazon'
DIR_VAL_CALTECH = 'data/office_caltech_10/VAL/caltech'
DIR_VAL_DSLR = 'data/office_caltech_10/VAL/dslr'
DIR_VAL_WEBCAM = 'data/office_caltech_10/VAL/webcam'
DIR_TEST = 'data/office_caltech_10/TEST'

# Define Data Preprocessing

# means and standard deviations ImageNet because the network is pretrained
# means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def office_caltech_10(transform=None):
    # transf = transforms.Compose([  # transforms.Resize(227),      # Resizes short size of the PIL image to 256
    #     transforms.CenterCrop(224),
    #     # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
    #     transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
    #     transforms.Normalize(means, stds)  # Normalizes tensor with mean and standard deviation
    # ])
    transf = transforms.Compose([  # 按照tinyimagenet的预处理方式处理，与我们的方法有些不同
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds)  # Normalizes tensor with mean and standard deviation
    ])

    transf_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds),
    ])
    if transform is None:
        transform = transf

    # Prepare Pytorch train/test Datasets
    amazon_dataset = torchvision.datasets.ImageFolder(DIR_AMAZON, transform=transform)
    caltech_dataset = torchvision.datasets.ImageFolder(DIR_CALTECH, transform=transform)
    dslr_dataset = torchvision.datasets.ImageFolder(DIR_DSLR, transform=transform)
    webcam_dataset = torchvision.datasets.ImageFolder(DIR_WEBCAM, transform=transform)

    val_amazon_dataset = torchvision.datasets.ImageFolder(DIR_VAL_AMAZON, transform=transf_test)
    val_caltech_dataset = torchvision.datasets.ImageFolder(DIR_VAL_CALTECH, transform=transf_test)
    val_dslr_dataset = torchvision.datasets.ImageFolder(DIR_VAL_DSLR, transform=transf_test)
    val_webcam_dataset = torchvision.datasets.ImageFolder(DIR_VAL_WEBCAM, transform=transf_test)

    test_dataset = torchvision.datasets.ImageFolder(DIR_TEST, transform=transf_test)

    # Check dataset sizes
    print(f"Amazon Dataset: {len(amazon_dataset)}")
    print(f"Caltech Dataset: {len(caltech_dataset)}")
    print(f"Dslr Dataset: {len(dslr_dataset)}")
    print(f"Webcam Dataset: {len(webcam_dataset)}")
    print(f"Val Amazon Dataset: {len(val_amazon_dataset)}")
    print(f"Val Caltech Dataset: {len(val_caltech_dataset)}")
    print(f"Val Dslr Dataset: {len(val_dslr_dataset)}")
    print(f"Val Webcam Dataset: {len(val_webcam_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")

    # print(photo_dataset.imgs)
    # print(photo_dataset.class_to_idx)
    #
    # BATCH_SIZE = 128
    # # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    # photo_dataloader = DataLoader(photo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    # art_dataloader = DataLoader(art_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    # cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    #                                 drop_last=False)
    # sketch_dataloader = DataLoader(sketch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    return amazon_dataset, caltech_dataset, dslr_dataset, webcam_dataset, val_amazon_dataset, val_caltech_dataset, val_dslr_dataset, val_webcam_dataset, test_dataset
