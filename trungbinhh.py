import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

response = requests.get('https://cdn2.fptshop.com.vn/unsafe/Uploads/images/tin-tuc/176652/Originals/noise-la-gi-2.jpg', stream=True)
response.raise_for_status()
image_data = BytesIO(response.content)

img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


plt.imshow(img, cmap='gray')


def trungbinh(img):
    for i in range(img.shape[0] - 2):
        for j in range(img.shape[1] - 2):
            img[i + 1, j + 1] = img[i:i + 3, j:j + 3].mean()
    return img


result = trungbinh(img)

plt.imshow(result, cmap='gray')