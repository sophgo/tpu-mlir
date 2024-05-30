import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_data(path: str = '', input_size: int = 224, batch_size: int = 64,
              num_workers: int = 4, pin_memory: bool = True, test_resize: int = 256):
    print('begin load datset')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(test_resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    print('finish load datset')
    return train_loader, val_loader
