# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2

name = "melon-2"
ext = ".jpg"

img_dir = "/workspace/_python/open-cv-python/img/"
dst_dir = img_dir + name + "/"

filename = name + ext

# 이미지 읽기
img = cv2.imread(img_dir + filename)

if img is None:
    print('Image load failed!')
    sys.exit()

src = img.copy()

# --- Pre-Processing: Blurring ---

# blurred = cv2.blur(mask, (3, 3))
blurred = cv2.bilateralFilter(src, -1, 10, 5)
cv2.imwrite(dst_dir + 'blurred.jpg', blurred)

# --- Edge Detection ---

grayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(grayscale, 50, 150, apertureSize=3)

# 팽창 후 침식
kernel = np.ones((7, 7), np.uint8)
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

cv2.imwrite(dst_dir + 'edge.jpg', edge)

# --- Find Contours ---

contours, _ = cv2.findContours(edge,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dst = src.copy()

for contour in contours:
    cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

    x, y, w, h = cv2.boundingRect(contour)
    if w / h > 0.8 and w / h < 1.2 and w > 60:
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(dst, str(w), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

cv2.imwrite(dst_dir + 'contours.jpg', dst)
