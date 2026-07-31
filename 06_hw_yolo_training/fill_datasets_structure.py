import cv2
import numpy as np
import random
from common import IMAGES_TRAIN, LABELS_TRAIN, IMAGES_VAL, LABELS_VAL, BASE_DIR

WIDTH = 256
HEIGHT = 256

NUM_TRAIN = 500
NUM_VAL = 100

CLASS_CIRCLE = 0
CLASS_SQUARE = 1

def make_img_circle(x_center, y_center, half_size, path):
    img_arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.circle(img_arr, (x_center, y_center), half_size, (255, 255, 255), -1)
    cv2.imwrite(path, img_arr)

def make_img_square(x_center, y_center, half_size, path):
    img_arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    top_left = (x_center - half_size, y_center - half_size)
    bottom_right = (x_center + half_size, y_center + half_size)
    cv2.rectangle(img_arr, top_left, bottom_right, (255, 255, 255), -1)
    cv2.imwrite(path, img_arr)

MAKE_IMG_BY_CLASS = {
    CLASS_CIRCLE: make_img_circle,
    CLASS_SQUARE: make_img_square,
}

def make_label(class_id, x_center, y_center, half_size, path):
    x = x_center / WIDTH
    y = y_center / HEIGHT
    w = half_size * 2 / WIDTH
    h = half_size * 2 / HEIGHT
    with open(path, 'w') as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

def random_shape_params():
    half_size = random.randint(1, min(WIDTH, HEIGHT) // 2)
    x_center = random.randint(half_size, WIDTH - half_size)
    y_center = random.randint(half_size, HEIGHT - half_size)
    return x_center, y_center, half_size

def generate_dataset(count, images_dir, labels_dir):
    class_ids = [CLASS_CIRCLE] * (count // 2) + [CLASS_SQUARE] * (count - count // 2)
    random.shuffle(class_ids)
    for i, class_id in enumerate(class_ids):
        x, y, half_size = random_shape_params()
        MAKE_IMG_BY_CLASS[class_id](x, y, half_size, images_dir + f'/img{i}.jpg')
        make_label(class_id, x, y, half_size, labels_dir + f'/img{i}.txt')

generate_dataset(NUM_TRAIN, BASE_DIR + IMAGES_TRAIN, BASE_DIR + LABELS_TRAIN)
generate_dataset(NUM_VAL, BASE_DIR + IMAGES_VAL, BASE_DIR + LABELS_VAL)
