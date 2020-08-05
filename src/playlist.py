import sys
import numpy as np
import cv2
from collections import Counter

img_dir = "/workspace/_python/open-cv-python/img/"
dst_dir = img_dir + "playlist/"

filename = 'genie.jpg'

# 이미지 읽기
img = cv2.imread(img_dir + filename)

if img is None:
    print('Image load failed!')
    sys.exit()

src = img.copy()

# src = cv2.bilateralFilter(src, -1, 10, 5)
src = cv2.medianBlur(src, 7)

# --- inrange_hue ---

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(src_hsv)

h_counter = Counter(h[:, 3])
s_counter = Counter(s[:, 3])
v_counter = Counter(v[:, 3])

h_mode = h_counter.most_common(1)[0][0]
s_mode = s_counter.most_common(1)[0][0]
v_mode = v_counter.most_common(1)[0][0]

mode = np.uint8([h_mode, s_mode, v_mode])

mask = cv2.inRange(src_hsv, mode, mode)
cv2.imwrite(dst_dir + 'mask.jpg', mask)

# --- Pre-Processing: Blurring ---

blurred = cv2.blur(mask, (3, 3))
cv2.imwrite(dst_dir + 'blurred.jpg', blurred)

# --- Edge Detection ---

# grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(blurred, 50, 150)
cv2.imwrite(dst_dir + 'edge.jpg', edge)

# --- Find Contours ---

contours, _ = cv2.findContours(edge,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dst = src.copy()

for contour in contours:
    # cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

    x, y, w, h = cv2.boundingRect(contour)
    if w / h > 0.8 and w / h < 1.2 and w > 55:
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # cv2.putText(dst, str(w), (x, y),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

cv2.imwrite(dst_dir + 'contours.jpg', dst)
