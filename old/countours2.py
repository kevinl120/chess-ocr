import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

img_num = random.randint(0, 99)
# img_num = 66
# img_num = 55
# img_num = 45
# img_num = 62
print(img_num)
img = cv2.imread(f'train/imgs/img_{img_num:04d}.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# tmp = cv2.blur(gray, (7, 7))
# tmp = cv2.GaussianBlur(gray, (5, 5), 0)
# tmp = cv2.bilateralFilter(gray, 3, 75, 75)
thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)[1]
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
tmp = 255 - cv2.morphologyEx(255 - th2, cv2.MORPH_CLOSE, np.ones((1, 1)))
black = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY)[1]

_, contours, _ = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

black = cv2.cvtColor(black, cv2.COLOR_GRAY2RGB)
for cnt in contours:
    area = cv2.contourArea(cnt)
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # if (0.01 * 750 * 500) < area:
    if len(approx) == 4 and (0.01 * 750 * 500) < area < (0.9 * 750 * 500):
        # img = cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)
        # black = cv2.drawContours(black, [cnt], 0, (0, 0, 255), 1)
        #
        x, y, w, h = cv2.boundingRect(cnt)
        black = cv2.rectangle(black, (x, y), (x + w, y + h), (0, 0, 255), 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('thresh', img)
cv2.waitKey()
# cv2.imshow('thresh', th2)
# cv2.waitKey()
# cv2.imshow('tmp', tmp)
# cv2.waitKey()
# cv2.imshow('tmp', black)
# cv2.waitKey()

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show(block=False)
# plt.imshow(thresh)
