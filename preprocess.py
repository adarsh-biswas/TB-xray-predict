# preprocess.py

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Constants
IMG_SIZE = 224  # EfficientNetB0 expects 224x224
BATCH_SIZE = 32

def get_dataloaders(data_dir='tb_dataset'):
    """
    Returns PyTorch DataLoaders for training, validation, and test sets.
    """

    # EfficientNetB0 normalization values (ImageNet mean & std)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Datasets
    train_dataset = ImageFolder(f'{data_dir}/train', transform=transform_train)
    val_dataset   = ImageFolder(f'{data_dir}/val', transform=transform_eval)
    test_dataset  = ImageFolder(f'{data_dir}/test', transform=transform_eval)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes
