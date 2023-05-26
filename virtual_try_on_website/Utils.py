import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile


def read_base64(image_base64, mode='cv2', grayscale=False):
    base64_data = image_base64.split(',')[1]
    if mode == 'cv2':
        nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
        if grayscale:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        buffer = BytesIO(base64.b64decode(base64_data))
        img = Image.open(buffer)
    return img


def base64_to_pil(image_base64):
    base64_data = image_base64.split(',')[1]
    buffer = BytesIO(base64.b64decode(base64_data))
    return Image.open(buffer)


def write_base64(img, format='.jpg'):
    _, buffer = cv2.imencode(format, img)
    encoded_data = base64.b64encode(buffer)
    return encoded_data


def read_uploaded_image(img: InMemoryUploadedFile, color='rgb'):
    img_bytes = img.read()
    nparr = np.fromstring(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if color == 'bgr':
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
