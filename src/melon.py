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

height, width, _ = img.shape

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

dst = np.zeros((height, width, 1), dtype=np.uint8)

# --- 경계 사각형 찾기 ---

rects = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    rects.append([x, y, w, h])
    cv2.rectangle(dst, (x, y), (x+w, y+h), (255, 255, 255), 2)

cv2.imwrite(dst_dir + 'rects.jpg', dst)

# --- Find Lines ---

lines = cv2.HoughLines(dst, 1, np.pi/180, 180)

dst = src.copy()

temp_x = []

for line in lines:
    for rho, theta in line:
        if theta == 0.0:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0+1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            temp_x.append(x1)

            cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(dst, str(theta), (x0, y0),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

temp_x.sort()

MOE = width / 100

temp_y = []
temp_h = []

first = temp_x[0]
temp_x = temp_x[1:]
for x in temp_x:
    if x - first > MOE:
        second = x
        break
album_w = abs(first - second)
print(album_w)

cv2.line(dst, (first, 0), (first, height), (0, 255, 0), 2)
cv2.line(dst, (second, 0), (second, height), (0, 255, 0), 2)

cv2.imwrite(dst_dir + 'lines.jpg', dst)

# --- Find Albums ---

dst = src.copy()

for rect in rects:
    # cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

    x, y, w, h = rect
    if abs(x - first) < MOE and abs(w - album_w) < MOE and abs(h - album_w) < MOE:
        temp_y.append(y)
        temp_h.append(y + h)
        cv2.rectangle(dst, (first, y), (second, y+h), (0, 0, 255), 2)
        cv2.putText(dst, str(w), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

if len(temp_y):
    temp_y.sort()
    temp_h.sort()

    gap = -1

    for idx, y in enumerate(temp_y[1:]):
        if gap < 0 and y - temp_h[idx] < album_w:
            gap = y - temp_h[idx]
            break

    for idx, y in enumerate(temp_y[1:]):
        if gap > 0:
            if y - temp_h[idx] > album_w:
                top = temp_h[idx] + gap
                bottom = top + album_w
                while True:
                    cv2.rectangle(dst, (first, top), (second, bottom), (0, 0, 255), 2)
                    if y - bottom > album_w:
                        top = bottom + gap
                        bottom = top + album_w
                        continue
                    break

cv2.imwrite(dst_dir + 'contours.jpg', dst)
