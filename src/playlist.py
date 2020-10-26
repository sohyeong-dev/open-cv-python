# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2
from collections import Counter

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024  # 1MB
LIMIT_BOX = 40

name = "black"
ext = ".jpg"

img_dir = "/workspace/_python/open-cv-python/img/"
# dst_dir = img_dir + "playlist/"
dst_dir = img_dir + name + "/"

# filename = 'genie.jpg'
filename = name + ext

# 이미지 읽기
img = cv2.imread(img_dir + filename)

if img is None:
    print('Image load failed!')
    sys.exit()

# --- resize ---

height, width, _ = img.shape
print(height, width)

# 오차 범위
MOE = width / 100

# if LIMIT_PX < height or LIMIT_PX < width:
#     ratio = float(LIMIT_PX) / max(height, width)
#     img = cv2.resize(img, None, fx=ratio, fy=ratio)
#     height, width, _ = height, width, _ = img.shape

src = img.copy()

# --- inrange_hue ---    1

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(src_hsv)

half = int(width / 2)

h_counter = Counter(h[:, half])
s_counter = Counter(s[:, half])
v_counter = Counter(v[:, half])

h_mode = h_counter.most_common(1)[0][0]
s_mode = s_counter.most_common(1)[0][0]
v_mode = v_counter.most_common(1)[0][0]

mode = np.uint8([h_mode, s_mode, v_mode])

mask = cv2.inRange(src_hsv, mode, mode)

# --- 자르기 ---

if mask[int(height / 2), 0] == 0:
    for idx, bgr in enumerate(mask[int(height / 2), 1:]):
        if bgr == 255:
            mask = mask[:, idx + 2:]
            src = src[:, idx + 2:]
            img = img[:, idx + 2:]
            break

cv2.imwrite(dst_dir + 'ori-mask.jpg', mask)

# 팽창 후 침식    2
kernel = np.ones((5, 5), np.uint8)
blurred = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 침식 후 팽창 - 앨범 이미지에 배경색과 같은 색이 있을 경우 대비
kernel = np.ones((3, 5), np.uint8)
blurred = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

cv2.imwrite(dst_dir + 'mask.jpg', blurred)

# --- Edge Detection ---    3

# grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(blurred, 50, 150, apertureSize=3)
cv2.imwrite(dst_dir + 'edge.jpg', edge)

# --- Find Rectangle ---    4

dst = src.copy()

lines = cv2.HoughLines(edge, 1, np.pi/180, 180)

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
temp_y = []
temp_h = []
# --- 결과 없으면 다음 직선으로 반복 ---    b5
while len(temp_y) == 0 and len(temp_x) > 1:
    first = temp_x[0]
    temp_x = temp_x[1:]
    for x in temp_x:
        if x - first > MOE:
            second = x
            break
    album_w = abs(first - second)
    print(album_w)

    cv2.imwrite(dst_dir + 'lines.jpg', dst)

    # --- Find Contours ---    5

    contours, _ = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dst = src.copy()

    temp_y = []

    cnt = 1

    for contour in contours:
        # cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

        # --- 경계 사각형 찾기 ---
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 5)
        # if w / h > 0.8 and w / h < 1.2 and w > 55:
        # --- 앨범 이미지 판단 조건 ---
        if abs(x - first) < MOE and abs(w - album_w) < MOE and abs(h - album_w) < MOE:    # 6
            temp_y.append(y)
            temp_h.append(y + h)
            # cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(dst, (first, y), (second, y+h), (0, 0, 255), 2)
            # cv2.putText(dst, str(w), (x, y),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            # --- 앨범 이미지 영역 ---
            roi = img[y:y+h, first:second]
            cv2.imwrite(dst_dir + 'albums/' + str(cnt).zfill(2) + '.jpg', roi)
            # --- 곡, 아티스트명 문자 영역 ---
            roi = img[y:y+h, second:]
            cv2.imwrite(dst_dir + 'right-boxes/' + str(cnt).zfill(2) + '.jpg', roi)
            cnt += 1

    # --- 앨범 이미지에 배경색과 같은 색이 있을 경우 대비 ---

    edge2 = cv2.Canny(mask, 50, 150, apertureSize=3)
    cv2.imwrite(dst_dir + 'ori-edge.jpg', edge2)

    contours, _ = cv2.findContours(edge2,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

        x, y, w, h = cv2.boundingRect(contour)
        is_added = False
        # cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 5)
        # if w / h > 0.8 and w / h < 1.2 and w > 55:
        if abs(x - first) < MOE and abs(w - album_w) < MOE and abs(h - album_w) < MOE:    # 6
            # --- 이미 추가됐는지 확인 ---
            for t in temp_y:
                if abs(y - t) < album_w:
                    is_added = True
                    break

            # --- 없으면 추가 ---
            if not is_added:
                temp_y.append(y)
                temp_h.append(y + h)
                # cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(dst, (first, y), (second, y+h), (0, 0, 255), 2)
                # cv2.putText(dst, str(w), (x, y),
                #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                roi = img[y:y+h, first:second]
                cv2.imwrite(dst_dir + 'albums/' + str(cnt).zfill(2) + '.jpg', roi)
                roi = img[y:y+h, second:]
                cv2.imwrite(dst_dir + 'right-boxes/' + str(cnt).zfill(2) + '.jpg', roi)
                cnt += 1

if len(temp_y):
    temp_y.sort()
    temp_h.sort()

    gap = -1

    for idx, y in enumerate(temp_y[1:]):
        if gap < 0 and y - temp_h[idx] < album_w:
            gap = y - temp_h[idx]
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
print(cnt - 1)
