import cv2
import numpy as np
import operator
import pdb

from helpers import *

def preprocess(img):
    res = img.copy()
    res = cv2.GaussianBlur(res, (9, 9), 0)
    # res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, res = cv2.threshold(res, 240, 255, cv2.THRESH_BINARY)
    # res = cv2.bitwise_not(res, res)
    return res

def find_boards(img, min_area=1000, square_thresh=10):
    """Finds the 4 corners of all square contours in the image."""
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending

    boards = []

    for polygon in contours:
        if cv2.contourArea(polygon) < min_area:
            break

        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        if abs(abs(polygon[top_left,0,1] - polygon[bottom_left,0,1]) - abs(polygon[top_left,0,0] - polygon[top_right,0,0])) < square_thresh:
            boards.append([polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]])
    
    return boards


def infer_grid(img):
	"""Infers 64 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 8
	for i in range(8):
		for j in range(8):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares

def main():
    img = cv2.imread('silman.png', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    # processed = preprocess(img)
    show_image(img)

    # bgr = processed
    # bgr = cv2.resize(bgr, (bgr.shape[1]//2, bgr.shape[0]//2))

    # lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # _, corners = cv2.findChessboardCorners(bgr, (7,7))
    # corners = np.int0(corners).reshape(-1, 2)
    # display_points(bgr, corners)
    # show_image(bgr)


    # processed = preprocess(img)
    # corners = find_boards(processed)
    # flattened = []
    # for x in corners:
    #     for y in x:
    #         flattened.append(y)
    # display_points(processed, flattened, radius=5)

    # for b in corners:
    #     cropped = img[b[0][1]:b[2][1], b[0][0]:b[2][0]]
    #     # show_image(cropped)
    #     squares = infer_grid(cropped)
    #     display_rects(cropped, squares)



if __name__ == '__main__':
    main()