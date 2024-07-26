import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

NUM_CLASSES = 7  # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']

DIR_PHOTO = 'Homework3-PACS/PACS/photo'
DIR_ART = 'Homework3-PACS/PACS/art_painting'
DIR_CARTOON = 'Homework3-PACS/PACS/cartoon'
DIR_SKETCH = 'Homework3-PACS/PACS/sketch'
DIR_VAL_PHOTO = 'Homework3-PACS/VAL/photo'
DIR_VAL_ART = 'Homework3-PACS/VAL/art_painting'
DIR_VAL_CARTOON = 'Homework3-PACS/VAL/cartoon'
DIR_VAL_SKETCH = 'Homework3-PACS/VAL/sketch'
DIR_TEST = 'Homework3-PACS/TEST'

# Define Data Preprocessing

# means and standard deviations ImageNet because the network is pretrained
means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pacs(transform=None):
    # transf = transforms.Compose([  # transforms.Resize(227),      # Resizes short size of the PIL image to 256
    #     transforms.CenterCrop(224),
    #     # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
    #     transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
    #     transforms.Normalize(means, stds)  # Normalizes tensor with mean and standard deviation
    # ])
    transf = transforms.Compose([  # 按照tinyimagenet的预处理方式处理，与我们的方法有些不同
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)  # Normalizes tensor with mean and standard deviation
    ])
    transf_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    if transform is None:
        transform = transf

    # Prepare Pytorch train/test Datasets
    photo_dataset = torchvision.datasets.ImageFolder(DIR_PHOTO, transform=transform)
    art_dataset = torchvision.datasets.ImageFolder(DIR_ART, transform=transform)
    cartoon_dataset = torchvision.datasets.ImageFolder(DIR_CARTOON, transform=transform)
    sketch_dataset = torchvision.datasets.ImageFolder(DIR_SKETCH, transform=transform)

    val_photo_dataset = torchvision.datasets.ImageFolder(DIR_VAL_PHOTO, transform=transf_test)
    val_art_dataset = torchvision.datasets.ImageFolder(DIR_VAL_ART, transform=transf_test)
    val_cartoon_dataset = torchvision.datasets.ImageFolder(DIR_VAL_CARTOON, transform=transf_test)
    val_sketch_dataset = torchvision.datasets.ImageFolder(DIR_VAL_SKETCH, transform=transf_test)

    test_dataset = torchvision.datasets.ImageFolder(DIR_TEST, transform=transf_test)

    # Check dataset sizes
    print(f"Photo Dataset: {len(photo_dataset)}")
    print(f"Art Dataset: {len(art_dataset)}")
    print(f"Cartoon Dataset: {len(cartoon_dataset)}")
    print(f"Sketch Dataset: {len(sketch_dataset)}")
    print(f"Val Photo Dataset: {len(val_photo_dataset)}")
    print(f"Val Art Dataset: {len(val_art_dataset)}")
    print(f"Val Cartoon Dataset: {len(val_cartoon_dataset)}")
    print(f"Val Sketch Dataset: {len(val_sketch_dataset)}")
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

    return photo_dataset, art_dataset, cartoon_dataset, sketch_dataset, val_photo_dataset, val_art_dataset, val_cartoon_dataset, val_sketch_dataset, test_dataset
