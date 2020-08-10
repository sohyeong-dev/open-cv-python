#-*- coding:utf-8-*-

import json

import requests
import sys
import numpy as np
import cv2
from collections import Counter

LIMIT_PX = 1024
LIMIT_BYTE = 1024*1024  # 1MB
LIMIT_BOX = 40

img_dir = "/workspace/_python/open-cv-python/img/"
dst_dir = img_dir + "playlist/"

filename = 'genie.jpg'

# 이미지 읽기
img = cv2.imread(img_dir + filename)

if img is None:
    print('Image load failed!')
    sys.exit()

# --- resize ---

height, width, _ = img.shape
print(height, width)

if LIMIT_PX < height or LIMIT_PX < width:
    ratio = float(LIMIT_PX) / max(height, width)
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    height, width, _ = height, width, _ = img.shape

src = img.copy()

# --- color_equalize ---

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

ycrcb_planes = cv2.split(src_ycrcb)

ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])

equalize_ycrcb = cv2.merge(ycrcb_planes)

equalize = cv2.cvtColor(equalize_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(dst_dir + 'equalize.jpg', equalize)

# src = cv2.bilateralFilter(src, -1, 10, 5)
blurred = cv2.medianBlur(equalize, 7)

# --- inrange_hue ---

src_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

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

# blurred = cv2.blur(mask, (3, 3))
# cv2.imwrite(dst_dir + 'blurred.jpg', blurred)

# --- Edge Detection ---

# grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(mask, 50, 150, apertureSize=3)
cv2.imwrite(dst_dir + 'edge.jpg', edge)

# --- Find Rectangle ---

dst = src.copy()

lines = cv2.HoughLines(edge,1,np.pi/180,100)

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
            y2 = int(y0 -1000*(a))
            
            temp_x.append(x1)

            cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(dst, str(theta), (x0, y0),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            
temp_x.sort()
album_w = abs(temp_x[0] - temp_x[1])

cv2.imwrite(dst_dir + 'lines.jpg', dst)

dst = src.copy()

minLineLength = 100
maxLineGap = 0

lines = cv2.HoughLinesP(edge,1,np.pi/360,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),3)
        
cv2.imwrite(dst_dir + 'linesP.jpg', dst)

# --- Find Contours ---

contours, _ = cv2.findContours(edge,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dst = src.copy()

temp_y = []

for contour in contours:
    # cv2.drawContours(dst, [contour], 0, [255, 0, 0], 2)

    x, y, w, h = cv2.boundingRect(contour)
    # if w / h > 0.8 and w / h < 1.2 and w > album_w - 2:
    if x > temp_x[0] - 1 and x < temp_x[1] + 1 and h > album_w - 2:
        temp_y.append(y)
        # cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(dst, (temp_x[0], y), (temp_x[1], y+h), (0, 0, 255), 2)
        # cv2.putText(dst, str(w), (x, y),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        
temp_y.sort()

cv2.imwrite(dst_dir + 'contours.jpg', dst)

ocr_img = img.copy()
roi = img[temp_y[0]:temp_y[len(temp_y) - 1] + album_w, temp_x[1]:width]
ocr_img[temp_y[0]:temp_y[len(temp_y) - 1] + album_w, 0:width-temp_x[1]] = roi

cv2.imwrite(dst_dir + 'ocr_img.jpg', roi)

exit()

# --- kakao ocr detect ---

API_URL = 'https://kapi.kakao.com/v1/vision/text/detect'

headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

jpeg_image = cv2.imencode(".jpg", img)[1]
data = jpeg_image.tobytes()

output = requests.post(API_URL, headers=headers, files={"file": data}).json()

boxes = output["result"]["boxes"]
boxes = boxes[:min(len(boxes), LIMIT_BOX)]

# --- kakao ocr recognize ---

API_URL = 'https://kapi.kakao.com/v1/vision/text/recognize'

headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

jpeg_image = cv2.imencode(".jpg", img)[1]
data = jpeg_image.tobytes()

output = requests.post(API_URL, headers=headers, files={"file": data}, data={"boxes": json.dumps(boxes)}).json()
print("[recognize] output:\n{}\n".format(json.dumps(output, sort_keys=True, indent=2)))
