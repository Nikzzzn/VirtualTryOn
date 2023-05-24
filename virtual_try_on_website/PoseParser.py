import os.path

import cv2
import json
import numpy as np
from django.conf import settings


def pose_parse(person_image):
    # MODE = "COCO"
    protoFile = os.path.join(settings.STATICFILES_DIRS[0], "./pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(settings.STATICFILES_DIRS[0], "./pose_iter_440000.caffemodel")
    nPoints = 18
    # POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    frameWidth = person_image.shape[1]
    frameHeight = person_image.shape[0]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    inWidth = 192
    inHeight = 256
    inpBlob = cv2.dnn.blobFromImage(person_image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    a = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        a.append(x)
        a.append(y)
        a.append(prob)
    dicti = {"pose_keypoints": a}
    people = []
    people.append(dicti)
    dicti = {"people":people}

    return dicti