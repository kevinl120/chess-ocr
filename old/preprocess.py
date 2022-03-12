import cv2
import glob
import numpy as np
import pdf2image

from constants import *


def pdfs_to_imgs(pdf_dir=PDF_DIR, img_dir=IMG_DIR):
    print('Converting PDFs to PNGs...')
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))
    i = 0
    for path in pdf_paths[6:7]:
        print(f'\tConverting {path}...')
        pil_imgs = pdf2image.convert_from_path(path)
        for img in pil_imgs:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)  # convert from PIL img to cv2 grayscale
            if img.shape[1] > img.shape[0]:  # the image may be two pages side by side
                img1, img2 = img[:, :img.shape[1]//2], img[:, img.shape[1]//2:]
                img1 = cv2.resize(img1, (IMG_WIDTH, IMG_LEN), interpolation=cv2.INTER_NEAREST)
                img2 = cv2.resize(img2, (IMG_WIDTH, IMG_LEN), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img1)
                i += 1
                cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img2)
            else:
                img = cv2.resize(img, (IMG_WIDTH, IMG_LEN), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(img_dir, f'img_{i:04d}.png'), img)
            i += 1
    print('Finished PDF to PNG conversion.')


def main():
    pdfs_to_imgs(pdf_dir='test/pdfs/', img_dir='imgs2/')


if __name__ == '__main__':
    main()
