import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 100
OUTPUT_PATH = 'saved_images'
ART_IMAGE_PATH = 'data/artworks/images/images/Vincent_van_Gogh'
IMAGE_PATH = 'data/flickr/Images'

TRANSFORM = A.Compose(
    [
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={'image0': 'image'}
)

BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0 #can be define to preserve color

CHECKPOINT_GENERATOR_I = "genh.pth.tar"
CHECKPOINT_GENERATOR_A = "genz.pth.tar"
CHECKPOINT_DISCRIMINATOR_I = "critich.pth.tar"
CHECKPOINT_DISCRIMINATOR_A = "criticz.pth.tar"