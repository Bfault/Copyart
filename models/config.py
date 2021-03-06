import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

ARTISTS = ['Cezanne', 'Monet', 'Ukiyoe', 'Van Gogh']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 100
OUTPUT_PATH = './saved_images'
ARTIST = 'Vincent_van_Gogh'
ART_IMAGE_PATH = f'./datasets/artworks/images/images/{ARTIST}'
IMAGE_PATH = './datasets/flickr/Images'
RESOLUTION = 128
PROD_RESOLUTION = 1024

TRANSFORM = A.Compose(
    [
        A.Resize(RESOLUTION, RESOLUTION),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={'image0': 'image'}
)

PROD_TRANSFORM = A.Compose(
    [
        A.Resize(PROD_RESOLUTION, PROD_RESOLUTION),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={'image0': 'image'}
)

BATCH_SIZE = 1
NUM_EPOCHS = 100
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0 #can be define to preserve color

CHECKPOINT_GENERATOR_I = "./checkpoints/generators/" + ARTIST + "/images/{}.pth"
CHECKPOINT_GENERATOR_A = "./checkpoints/generators/" + ARTIST + "/artworks/{}.pth"
CHECKPOINT_DISCRIMINATOR_I = "./checkpoints/discriminators/" + ARTIST + "/images/{}.pth"
CHECKPOINT_DISCRIMINATOR_A = "./checkpoints/discriminators/" + ARTIST + "/artworks/{}.pth"