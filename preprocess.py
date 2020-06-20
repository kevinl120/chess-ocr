import cv2
import glob
import os
import pdf2image

from constants import *

def resize(img_dir=IMG_DIR, small_img_dir=SMALL_IMG_DIR):
    os.makedirs(small_img_dir, exist_ok=True)
    img_files = glob.glob(os.path.join(img_dir, '*'))
    img_files.sort()
    for f in img_files:
        img = cv2.imread(f)
        img = cv2.resize(img, (SMALL_IMG_WIDTH, SMALL_IMG_LEN), interpolation=cv2.INTER_AREA)
        filename = f.split('/')[-1]
        cv2.imwrite(os.path.join(small_img_dir, filename), img)


def pdf_to_image(pdf_path, img_dir=IMG_DIR):
    pages = pdf2image.convert_from_path(pdf_path)
    for i, page in enumerate(pages):
        page.save(os.path.join(img_dir, 'img_{:04d}.png'.format(i)), 'PNG')


def main():
    pdf_to_image('./data/pdf.pdf')
    resize()


if __name__ == '__main__':
    main()