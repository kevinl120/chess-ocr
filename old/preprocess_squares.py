import cv2
import glob
import numpy as np
import os
import xml.etree.ElementTree as ET


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        filename = root.find('filename').text
        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return filename, list_with_all_boxes

def imgs_to_squares(train_dir='./train', out_dir='./squares_train/imgs'):
    print('Converting boards to squares')
    img_paths = sorted(glob.glob(os.path.join(train_dir, 'imgs', '*.png')))
    ann_paths = sorted(glob.glob(os.path.join(train_dir, 'annotations', '*.xml')))

    i = 0
    # split image into 64 squares
    for img_path, ann_path in zip(img_paths, ann_paths):
        print(img_path)
        img = cv2.imread(img_path)
        filename, list_with_all_boxes = read_content(ann_path)
        squares = []
        for box in list_with_all_boxes:
            xmin, ymin, xmax, ymax = box
            for x in range(xmin, xmax, (xmax - xmin) // 8):
                for y in range(ymin, ymax, (ymax - ymin) // 8):
                    if i > 10:
                        break
                    cv2.imwrite(os.path.join(out_dir, f'img_{i:04d}.png'), img[y:y + (ymax - ymin) // 8, x:x + (xmax - xmin) // 8])
                    i += 1

    print('Finished imgs to squares conversion')


def main():
    imgs_to_squares()


if __name__ == '__main__':
    main()
