import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for 32x32 CIFAR-10 images.
    """
    from torchvision import models

    model = models.resnet18(weights=None)
    # adapt first conv + remove maxpool for small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_cifar10_loaders(
    batch_size_train: int = 128,
    batch_size_test: int = 64,
    data_root: str = "./data",
):
    """
    Returns train, test and explain (batch_size=1) loaders for CIFAR-10.
    """
    transform_train = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )
    explain_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False
    )

    classes = train_set.classes

    return train_loader, test_loader, explain_loader, classes
