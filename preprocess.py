import cv2
import glob
import numpy as np
import pdf2image

from constants import *


def pdfs_to_imgs(pdf_dir=PDF_DIR, img_dir=IMG_DIR):
    print('Converting PDFs to PNGs...')
    pdf_paths = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    i = len(glob.glob(os.path.join(img_dir, '*.png')))
    for path in pdf_paths:
        print(f'\tConverting {path}...')
        pil_imgs = pdf2image.convert_from_path(path)
        for img in pil_imgs:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # convert from PIL img to cv2
            img = cv2.resize(img, (IMG_WIDTH, IMG_LEN), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img)
            i += 1
    print('Finished PDF to PNG conversion.')


def main():
    pdfs_to_imgs()


if __name__ == '__main__':
    main()
