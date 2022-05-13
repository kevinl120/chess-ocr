import cv2
import glob
import pdf2image
import xml.etree.ElementTree as ET

from constants import *

def convert_pdfs_to_imgs(pdf_dir, img_dir):
    print(f'Converting PDFs in {pdf_dir} to PNGs...')
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))
    img_num = 0
    for pdf_num, path in enumerate(pdf_paths):
        print(f'\t{pdf_num+1}/{len(pdf_paths)}: Converting {path}...')
        pil_imgs = pdf2image.convert_from_path(path, size=(IMG_WIDTH, None), grayscale=True)
        for img in pil_imgs:
            for y in range(0, img.size[1], IMG_LEN // 2):
                img_crop = img.crop((0, y, 0+IMG_LEN, y+IMG_LEN))
                img_crop.save(os.path.join(img_dir, f'img_{img_num:04d}.png'))
                img_num += 1
    print(f'Finished PDF to PNG conversion for {pdf_dir}. {img_num} images created.')

def convert_board_imgs_to_square_imgs(board_image_dir, board_annotation_dir, square_image_dir):
    print(f'Converting board images to square images...')
    board_img_paths = sorted(glob.glob(os.path.join(board_image_dir, '*.png')))
    board_annotation_paths = sorted(glob.glob(os.path.join(board_annotation_dir, '*.xml')))
    square_img_num = 0
    for img_num, (board_img_path, board_annotation_path) in enumerate(zip(board_img_paths, board_annotation_paths)):
        print(f'\t{img_num+1}/{len(board_img_paths)}: Converting {board_img_path}...')
        board_img = cv2.imread(board_img_path, cv2.IMREAD_GRAYSCALE)
        for xmin, ymin, xmax, ymax in get_bounding_boxes(board_annotation_path, board_img_path):
            board_resized = cv2.resize(board_img[ymin:ymax, xmin:xmax], (SQUARE_IMG_WIDTH * 8, SQUARE_IMG_LENGTH * 8), interpolation=cv2.INTER_AREA)
            for y in range(8):
                for x in range(8):
                    square_img = board_resized[y * SQUARE_IMG_WIDTH : (y+1) * SQUARE_IMG_WIDTH, x * SQUARE_IMG_LENGTH : (x+1) * SQUARE_IMG_LENGTH]
                    cv2.imwrite(os.path.join(square_image_dir, f'img_{square_img_num:05d}.png'), square_img)
                    square_img_num += 1
    print(f'Finished board image to square image conversion. {square_img_num} images created.')

def get_bounding_boxes(annotation_file_path, image_file_path):
    tree = ET.parse(annotation_file_path)
    root = tree.getroot()
    all_bounding_boxes = []
    for boxes in root.iter('object'):
        filename = root.find('filename').text
        if filename != image_file_path.split('/')[-1].split('.')[0]:
            raise Exception(f'Filename in annotation file {annotation_file_path} does not match image file {image_file_path}')
        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        all_bounding_boxes.append([xmin, ymin, xmax, ymax])
    return all_bounding_boxes

def main():
    # convert_pdfs_to_imgs(SEGMENTATION_PDF_DIR, SEGMENTATION_IMAGE_DIR)
    convert_board_imgs_to_square_imgs(SEGMENTATION_IMAGE_DIR, SEGMENTATION_ANNOTATION_DIR, CLASSIFICATION_IMAGE_DIR)
    pass

if __name__ == '__main__':
    main()