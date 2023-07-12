import io
import math
import os.path
import cv2
import numpy as np
from PIL import Image
from django.conf import settings


class OpenPose_Processor:
    POSE_BODY_25_COLORS = [
        255, 0, 85,
        255, 0, 0,
        255, 85, 0,
        255, 170, 0,
        255, 255, 0,
        170, 255, 0,
        85, 255, 0,
        0, 255, 0,
        255, 0, 0,
        0, 255, 85,
        0, 255, 170,
        0, 255, 255,
        0, 170, 255,
        0, 85, 255,
        0, 0, 255,
        255, 0, 170,
        170, 0, 255,
        255, 0, 255,
        85, 0, 255,
        0, 0, 255,
        0, 0, 255,
        0, 0, 255,
        0, 255, 255,
        0, 255, 255,
        0, 255, 255
    ]
    POSE_COCO_COLORS = [
        255, 0, 85,
        255, 0, 0,
        255, 85, 0,
        255, 170, 0,
        255, 255, 0,
        170, 255, 0,
        85, 255, 0,
        0, 255, 0,
        0, 255, 85,
        0, 255, 170,
        0, 255, 255,
        0, 170, 255,
        0, 85, 255,
        0, 0, 255,
        255, 0, 170,
        170, 0, 255,
        255, 0, 255,
        85, 0, 255
    ]
    POSE_BODY_25_PAIRS = [1,8, 1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 10,11, 8,12, 12,13, 13,14, 1,0, 0,15,
                          15,17, 0,16, 16,18, 14,19, 19,20, 14,21, 11,22, 22,23, 11,24]
    POSE_COCO_PAIRS = [1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17]

    def __init__(self):
        static_dir = settings.STATICFILES_DIRS[0]
        model_body25_proto = os.path.join(static_dir, "models", "openpose", "pose_deploy.prototxt")
        model_body25_weight = os.path.join(static_dir, "models", "openpose", "pose_iter_584000.caffemodel")
        model_coco_proto = os.path.join(static_dir, "models", "openpose", "pose_deploy_linevec.prototxt")
        model_coco_weight = os.path.join(static_dir, "models", "openpose", "pose_iter_440000.caffemodel")

        self.body25_model = cv2.dnn.readNetFromCaffe(model_body25_proto, model_body25_weight)
        self.coco_model = cv2.dnn.readNetFromCaffe(model_coco_proto, model_coco_weight)

    @staticmethod
    def get_rectangle(keypoints, threshold):
        minX = np.iinfo(int).max
        maxX = np.iinfo(int).min
        minY = minX
        maxY = maxX
        for i in range(25):
            score = keypoints[3 * i + 2]
            if score > threshold:
                x = keypoints[3 * i]
                y = keypoints[3 * i + 1]
                if maxX < x:
                    maxX = x
                if minX > x:
                    minX = x
                if maxY < y:
                    maxY = y
                if minY > y:
                    minY = y
        return [minX, minY, maxX-minX, maxY-minY]

    def pose_parse(self, person_image, n_points=25, width=192, height=256):
        if n_points == 18:
            model = self.coco_model
        elif n_points == 25:
            model = self.body25_model
            
        person_image = np.array(person_image)
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

    @staticmethod
    def resize_keypoints(pose_keypoints, old_size, new_size):
        width, height = old_size
        new_width, new_height = new_size

        width_ratio = new_width / width
        height_ratio = new_height / height

        for i in range(0, len(pose_keypoints), 3):
            pose_keypoints[i] = pose_keypoints[i] * width_ratio
            pose_keypoints[i + 1] = pose_keypoints[i + 1] * height_ratio

        return pose_keypoints

    @staticmethod
    def render_pose_skeleton(person_image, pose_keypoints, n_points=25):
        if n_points == 18:
            colors = OpenPose_Processor.POSE_COCO_COLORS
            pairs = OpenPose_Processor.POSE_COCO_PAIRS
        elif n_points == 25:
            colors = OpenPose_Processor.POSE_BODY_25_COLORS
            pairs = OpenPose_Processor.POSE_BODY_25_PAIRS

        line_type = 8
        shift = 0
        colors_number = len(colors)
        threshold = 0.1
        # threshold = 0
        thickness_circle_ratio = 1. / 75
        thickness_line_ratio_wrt_circle = 0.75
        keypoint_alpha = 0.6
        heatmap_alpha = 0.7

        person_image = np.array(person_image)
        height = person_image.shape[0]
        width = person_image.shape[1]
        area = width * height

        person_rectangle = OpenPose_Processor.get_rectangle(pose_keypoints, threshold=threshold)

        ratio_areas = min(1, max(person_rectangle[2] / width, person_rectangle[3] / height))
        thickness_ratio = max(math.sqrt(area) * thickness_circle_ratio * ratio_areas, 2)
        thickness_circle = max(1, round(thickness_ratio if ratio_areas > 0.05 else -1))
        thickness_line = max(1, round(thickness_ratio * thickness_line_ratio_wrt_circle))
        radius = round(thickness_ratio / 2)

        res_img = np.zeros((height, width, 3), dtype="uint8")

        for pair in range(0, len(pairs), 2):
            index1 = (0 * n_points + pairs[pair]) * 3
            index2 = (0 * n_points + pairs[pair + 1]) * 3
            if pose_keypoints[index1 + 2] > threshold and pose_keypoints[index2 + 2] > threshold:
                color_index = pairs[pair + 1] * 3
                color = (colors[(color_index + 2) % colors_number],
                         colors[(color_index + 1) % colors_number],
                         colors[color_index % colors_number])
                keypoint1 = (round(pose_keypoints[index1]), round(pose_keypoints[index1 + 1]))
                keypoint2 = (round(pose_keypoints[index2]), round(pose_keypoints[index2 + 1]))

                overlay = res_img.copy()
                cv2.line(overlay, keypoint1, keypoint2, color, thickness_line, line_type, shift)
                res_img = cv2.addWeighted(overlay, heatmap_alpha, res_img, 1 - heatmap_alpha, 0)

        for part in range(25):
            face_index = (0 + part) * 3
            if pose_keypoints[face_index + 2] > threshold:
                color_index = part * 3
                color = (colors[(color_index + 2) % colors_number],
                         colors[(color_index + 1) % colors_number],
                         colors[color_index % colors_number])
                center = (round(pose_keypoints[face_index]), round(pose_keypoints[face_index + 1]))
                overlay = res_img.copy()
                cv2.circle(overlay, center, radius, color, thickness_circle, line_type, shift)
                res_img = cv2.addWeighted(overlay, keypoint_alpha, res_img, 1 - keypoint_alpha, 0)

        _, buffer = cv2.imencode(".png", res_img)
        io_buf = io.BytesIO(buffer)
        res_img = Image.open(io_buf)

        return res_img
