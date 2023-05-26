import base64
import os
from io import BytesIO

import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.resnet import Bottleneck
from virtual_try_on_website import settings
from virtual_try_on_website.models.res_net import ResNet


class LIP_JPPNet:
    num_classes = 20
    input_size = (512, 512)

    def __init__(self):
        self.model = self.__load_model()

    @classmethod
    def __get_masking_palette(cls, num_cls):
        palette = [0] * (num_cls * 3)
        for j in range(0, num_cls):
            temp = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while temp:
                palette[j * 3 + 0] |= (((temp >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((temp >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((temp >> 2) & 1) << (7 - i))
                i += 1
                temp >>= 3
        return palette

    @classmethod
    def get_mask_arrays(cls, person_segmentation):
        shape = (person_segmentation > 0).astype(np.float32)
        head = (person_segmentation == 1).astype(np.float32) + \
               (person_segmentation == 2).astype(np.float32) + \
               (person_segmentation == 4).astype(np.float32) + \
               (person_segmentation == 13).astype(np.float32)
        head = (head > 0).astype(np.float32)
        cloth = (person_segmentation == 5).astype(np.float32) + \
                (person_segmentation == 6).astype(np.float32) + \
                (person_segmentation == 7).astype(np.float32)
        cloth = (cloth > 0).astype(np.float32)
        body = (person_segmentation == 1).astype(np.float32) + \
               (person_segmentation == 2).astype(np.float32) + \
               (person_segmentation == 3).astype(np.float32) + \
               (person_segmentation == 4).astype(np.float32) + \
               (person_segmentation > 7).astype(np.float32)
        body = (body > 0).astype(np.float32)
        return shape, head, cloth, body

    @classmethod
    def __load_model(cls):
        model = ResNet(Bottleneck, [3, 4, 23, 3], cls.num_classes)
        model.input_space = 'BGR'
        model.input_size = [3, 224, 224]
        model.input_range = [0, 1]
        model.mean = [0.406, 0.456, 0.485]
        model.std = [0.225, 0.224, 0.229]

        model = nn.DataParallel(model)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "./lip_jppnet.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @classmethod
    def __transform_image(cls, person_image):
        input_size = [512, 512]
        aspect_ratio = input_size[1] * 1.0 / input_size[0]
        person_center = np.zeros(2, dtype=np.float32)

        h, w, _ = person_image.shape
        person_center[0] = w * 0.5
        person_center[1] = h * 0.5
        # if w > aspect_ratio * h:
        #     h = w * 1.0 / aspect_ratio
        # elif w < aspect_ratio * h:
        #     w = h * aspect_ratio

        normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

        identity_transform = np.array([[1., -0., 0.], [0., 1., 0.]])
        image = cv2.warpAffine(
            person_image,
            identity_transform,
            (input_size[1], input_size[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        image = normalization_transform(image)

        return image

    @torch.no_grad()
    def parse_person(self, person_image):
        image = self.__transform_image(person_image)
        h = person_image.shape[0]
        w = person_image.shape[1]
        palette = self.__get_masking_palette(self.num_classes)

        output = self.model(image[None, :])
        upsample_output = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)(output)
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)
        upsample_output = upsample_output.data.cpu().numpy()

        identity_transform = np.array([[1., -0., 0.], [0., 1., 0.]])
        channel = upsample_output.shape[2]
        logits_result = []
        for i in range(channel):
            target_logit = cv2.warpAffine(
                upsample_output[:, :, i],
                identity_transform,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
            logits_result.append(target_logit)
        logits_result = np.stack(logits_result, axis=2)

        parsing_result = np.argmax(logits_result, axis=2)

        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_img.putpalette(palette)

        buffered = BytesIO()
        output_img.convert('P').save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("utf-8")
