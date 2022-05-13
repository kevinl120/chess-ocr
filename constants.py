import os

DATA_DIR = 'data'

SEGMENTATION_DIR = os.path.join(DATA_DIR, 'segmentation')
SEGMENTATION_PDF_DIR = os.path.join(SEGMENTATION_DIR, 'pdfs')
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_DIR, 'images')
SEGMENTATION_ANNOTATION_DIR = os.path.join(SEGMENTATION_DIR, 'annotations')

SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, 'imageai', 'train')
SEGMENTATION_VALIDATION_DIR = os.path.join(SEGMENTATION_DIR, 'imageai', 'validation')
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, 'imageai', 'test')

CLASSIFICATION_DIR = os.path.join(DATA_DIR, 'classification')
CLASSIFICATION_IMAGE_DIR = os.path.join(CLASSIFICATION_DIR, 'images')

IMG_WIDTH = 608
IMG_LEN = 608

SQUARE_IMG_WIDTH = 32
SQUARE_IMG_LENGTH = 32