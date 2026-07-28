import cv2
import numpy as np
import random

BASE_DIR = 'datasets/'
BASE_IMAGES = BASE_DIR + 'images/'
BASE_LABELS = BASE_DIR + 'labels/'
IMAGES_TRAIN = BASE_IMAGES + 'train/'
LABELS_TRAIN = BASE_LABELS + 'train/'
IMAGES_VAL = BASE_IMAGES + 'val/'
LABELS_VAL = BASE_LABELS + 'val/'

WIDTH = 256
HEIGHT = 256

NUM_TRAIN = 500
NUM_VAL = 100

def make_img_circle(x_center, y_center, radius, path):
    img_arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.circle(img_arr, (x_center, y_center), radius, (255, 255, 255), -1)
    cv2.imwrite(path, img_arr)

def make_label_circle(x_center, y_center, radius, path):
    x = x_center / WIDTH
    y = y_center / HEIGHT
    w = radius * 2 / WIDTH
    h = radius * 2 / HEIGHT
    with open(path, 'w') as f:
        f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

def random_circle_params():
    radius = random.randint(1, min(WIDTH, HEIGHT) // 2)
    x_center = random.randint(radius, WIDTH - radius)
    y_center = random.randint(radius, HEIGHT - radius)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return x_center, y_center, radius, color

def generate_dataset(count, images_dir, labels_dir):
    for i in range(count):
        x, y, radius, color = random_circle_params()
        make_img_circle(x, y, radius, images_dir + f'img{i}.jpg')
        make_label_circle(x, y, radius, labels_dir + f'img{i}.txt')

generate_dataset(NUM_TRAIN, IMAGES_TRAIN, LABELS_TRAIN)
generate_dataset(NUM_VAL, IMAGES_VAL, LABELS_VAL)
