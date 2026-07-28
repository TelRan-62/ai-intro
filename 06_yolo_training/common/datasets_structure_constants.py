BASE_DIR = 'datasets/'
BASE_IMAGES = 'images'
BASE_LABELS = 'labels'
IMAGES_TRAIN = BASE_IMAGES + '/train'
LABELS_TRAIN = BASE_LABELS + '/train'
IMAGES_VAL = BASE_IMAGES + '/val'
LABELS_VAL = BASE_LABELS + '/val'
DATA_YAML = BASE_DIR + 'data.yaml'
NAMES = {0: 'circle'}

__all__ = ['BASE_DIR', 'BASE_IMAGES', 'BASE_LABELS', 'IMAGES_TRAIN', 'LABELS_TRAIN', 'IMAGES_VAL', 'LABELS_VAL',
           'DATA_YAML', 'NAMES']
