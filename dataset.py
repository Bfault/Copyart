import os
from typing import Tuple

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ArtImageDataset(Dataset):
    def __init__(self, root_art: str, root_image: str, transform: any=None) -> None:
        self.root_art = root_art
        self.root_image = root_image
        self.transform = transform
        self.art_images = os.listdir(root_art)
        self.image_images = os.listdir(root_image)
        self.length_dataset = max(len(self.art_images), len(self.image_images))
        self.art_length = len(self.art_images)
        self.image_length = len(self.image_images)
    
    def __len__(self) -> int:
        return self.length_dataset
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        art_img = self.art_images[index % self.art_length]
        image_img = self.image_images[index % self.image_length]

        art_path = os.path.join(self.root_art, art_img)
        image_path = os.path.join(self.root_image, image_img)

        art_img = np.array(Image.open(art_path).convert('RGB'))
        image_img = np.array(Image.open(image_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=art_img, image0=image_img)
            art_img = augmentations['image']
            image_img = augmentations['image0']
        
        return art_img, image_img