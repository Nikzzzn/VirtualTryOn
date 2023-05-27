import os.path
import cv2
from django.conf import settings


class OpenPose_Processor:
    def __init__(self):
        static_dir = settings.STATICFILES_DIRS[0]
        model_coco_proto = os.path.join(static_dir, "openpose", "pose_deploy_linevec.prototxt")
        model_coco_weight = os.path.join(static_dir, "openpose", "pose_iter_440000.caffemodel")
        model_body25_proto = os.path.join(static_dir, "openpose", "pose_deploy.prototxt")
        model_body25_weight = os.path.join(static_dir, "openpose", "pose_iter_584000.caffemodel")

        self.coco_model = cv2.dnn.readNetFromCaffe(model_coco_proto, model_coco_weight)
        self.body25_model = cv2.dnn.readNetFromCaffe(model_body25_proto, model_body25_weight)


    def pose_parse(self, person_image, n_points=18, width=192, height=256):
        if n_points == 18:
            model = self.coco_model
        elif n_points == 25:
            model = self.body25_model

        frameWidth = person_image.shape[1]
        frameHeight = person_image.shape[0]
        inpBlob = cv2.dnn.blobFromImage(person_image, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
        model.setInput(inpBlob)
        output = model.forward()
        H = output.shape[2]
        W = output.shape[3]
        keypoints = []
        for i in range(n_points):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            keypoints.append(x)
            keypoints.append(y)
            keypoints.append(prob)

        return keypoints