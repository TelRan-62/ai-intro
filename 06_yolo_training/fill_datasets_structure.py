import cv2
import numpy as np

BASE_DIR = 'datasets/'
BASE_IMAGES = BASE_DIR + 'images/'
BASE_LABELS = BASE_DIR + 'labels/'
IMAGES_TRAIN = BASE_IMAGES + 'train/'
LABELS_TRAIN = BASE_LABELS + 'train/'
IMAGES_VAL = BASE_IMAGES + 'val/'
LABELS_VAL = BASE_LABELS + 'val/'

WIDTH = 256
HEIGHT = 256

def make_img_circle(x_center, y_center, radius, path):
    img_arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.circle(img_arr, (x_center, y_center), radius, (255, 255, 255), -1)
    cv2.imwrite(path, img_arr)

make_img_circle(100, 100, 30, IMAGES_TRAIN + 'img1.jpg')
make_img_circle(120, 70, 25, IMAGES_TRAIN + 'img2.jpg')
make_img_circle(90, 75, 20, IMAGES_TRAIN + 'img3.jpg')
make_img_circle(110, 95, 40, IMAGES_VAL + 'img1.jpg')

def make_label_circle(x_center, y_center, radius, path):
    x = x_center / WIDTH
    y = y_center / HEIGHT
    w = radius * 2 / WIDTH
    h = radius * 2 / HEIGHT
    with open(path, 'w') as f:
        f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

make_label_circle(100, 100, 30, LABELS_TRAIN + 'img1.txt')
make_label_circle(120, 70, 25, LABELS_TRAIN + 'img2.txt')
make_label_circle(90, 75, 20, LABELS_TRAIN + 'img3.txt')
make_label_circle(110, 95, 40, LABELS_VAL + 'img1.txt')
