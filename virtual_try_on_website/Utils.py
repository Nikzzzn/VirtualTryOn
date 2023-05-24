import base64
import cv2
import numpy as np


def readb64(image_uri):
    encoded_data = image_uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
