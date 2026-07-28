import os
import yaml
from common import IMAGES_TRAIN, LABELS_TRAIN, IMAGES_VAL, LABELS_VAL, DATA_YAML, NAMES, BASE_DIR


def make_data_yaml(**data):
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f)


os.makedirs(BASE_DIR + IMAGES_TRAIN, exist_ok=True)
os.makedirs(BASE_DIR + LABELS_TRAIN, exist_ok=True)
os.makedirs(BASE_DIR +IMAGES_VAL, exist_ok=True)
os.makedirs(BASE_DIR + LABELS_VAL, exist_ok=True)

make_data_yaml(
    path=BASE_DIR,
    train=IMAGES_TRAIN,
    val=IMAGES_VAL,
    names=NAMES
)

