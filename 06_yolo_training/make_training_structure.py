import os

BASE_DIR = 'datasets/'
BASE_IMAGES = BASE_DIR + 'images/'
BASE_LABELS = BASE_DIR + 'labels/'
IMAGES_TRAIN = BASE_IMAGES + 'train/'
LABELS_TRAIN = BASE_LABELS + 'train/'
IMAGES_VAL = BASE_IMAGES + 'val/'
LABELS_VAL = BASE_LABELS + 'val/'
os.makedirs(IMAGES_TRAIN, exist_ok=True)
os.makedirs(LABELS_TRAIN, exist_ok=True)
os.makedirs(IMAGES_VAL, exist_ok=True)
os.makedirs(LABELS_VAL, exist_ok=True)
