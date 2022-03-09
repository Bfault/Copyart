import numpy as np

import torch
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloader(path: str, image_size: int, batch_size: int, num_workers: int=4) -> DataLoader:

    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset: datasets.ImageFolder = datasets.ImageFolder(root=path, transform=transform)
    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader

def tensor_show(img: torch.Tensor) -> None:
    npimg = img.detach().numpy()
    npimg = npimg * 0.5 + 0.5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    batch_size, image_size = 5, 256
    train_loader = get_dataloader(path='./data/resized', image_size=image_size, batch_size=batch_size)
    dataiter = iter(train_loader)

    img,_ = next(dataiter)
    sample_img = img[-1]
    tensor_show(sample_img)