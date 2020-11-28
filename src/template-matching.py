# import the necessary packages
import numpy as np
from urllib import request
import cv2


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


src = cv2.imread("/workspace/_python/open-cv-python/img/flo-3/albums/01.jpg", cv2.IMREAD_COLOR)

# initialize the list of image URLs to download
urls = [
    "https://cdnimg.melon.co.kr/cm/album/images/021/75/668/2175668_500.jpg/melon/resize/144/optimize/90",
    "https://cdnimg.melon.co.kr/cm/album/images/101/70/914/10170914_500.jpg?3011bfc00533c569ae5794b26ee1ff23/melon/resize/144/optimize/90",
    "https://cdnimg.melon.co.kr/cm/album/images/100/03/272/10003272_500.jpg/melon/resize/144/optimize/90",
]

# loop over the image URLs
for url in urls:
    templit = url_to_image(url)

    h, w, _ = src.shape
    templit = cv2.resize(templit, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    result = cv2.matchTemplate(src, templit, cv2.TM_SQDIFF_NORMED)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc

    print(minVal)
