import base64
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from torchvision import transforms

from virtual_try_on_website import settings
from virtual_try_on_website.models.viton_hd import SegGenerator, VITON_HD_GMM, ALIASGenerator
from virtual_try_on_website.processors.lip_jppnet import LIP_JPPNet
from virtual_try_on_website.processors.openpose import OpenPose_Processor


class VITON_HD_Processor:
    def __init__(self):
        self.seg_model = self.__load_seg_model()
        self.gmm_model = self.__load_gmm_model()
        self.alias_model = self.__load_alias_model()
        self.fine_height = 1024
        self.fine_width = 768

    @staticmethod
    def __load_seg_model():
        model = SegGenerator(input_nc=13 + 8, output_nc=13)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/viton_hd/seg_final.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @staticmethod
    def __load_gmm_model():
        model = VITON_HD_GMM(inputA_nc=7, inputB_nc=3)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/viton_hd/gmm_final.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @staticmethod
    def __load_alias_model():
        model = ALIASGenerator(input_nc=9)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/viton_hd/alias_final.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @staticmethod
    def get_agnostic(person_image, person_segmentation, pose_data):
        parse_array = np.array(person_segmentation)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = person_image.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 7, pointy - r * 7, pointx + r * 7, pointy), 'gray', 'gray')
        agnostic.paste(person_image, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(person_image, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def prepare_data(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        semantic_nc = 13
        normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        person_image = Image.fromarray(person_image)

        cloth_image = Image.fromarray(cloth_image)
        cloth_image = transforms.Resize(self.fine_width, interpolation=2)(cloth_image)
        cloth_image = normalization_transform(cloth_image)

        cloth_mask = Image.fromarray(cloth_mask)
        cloth_mask = transforms.Resize(self.fine_width, interpolation=0)(cloth_mask)
        cloth_mask_array = np.array(cloth_mask)
        cloth_mask_array = (cloth_mask_array >= 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask_array)
        cloth_mask.unsqueeze_(0)

        if person_image.height != self.fine_height or person_image.width != self.fine_width:
            person_image = transforms.Resize(self.fine_width, interpolation=2)(person_image)
            pose_data = OpenPose_Processor().pose_parse(person_image)
            # pose_data = OpenPose_Processor.resize_keypoints(pose_data, person_image.size, (self.fine_width, self.fine_height))

        pose_data_reshaped = np.array(pose_data)
        pose_data_reshaped = pose_data_reshaped.reshape((-1, 3))[:, :2]

        person_segmentation = transforms.Resize(self.fine_width, interpolation=0)(person_segmentation)
        parse_agnostic = LIP_JPPNet.get_parse_agnostic(person_segmentation, pose_data_reshaped)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        person_image = transforms.Resize(self.fine_width, interpolation=2)(person_image)
        person_image_tensor = normalization_transform(person_image)

        agnostic = self.get_agnostic(person_image, person_segmentation, pose_data_reshaped)
        agnostic = transforms.Resize(self.fine_width, interpolation=0)(agnostic)
        agnostic = normalization_transform(agnostic)

        pose_skeleton = OpenPose_Processor.render_pose_skeleton(person_image, pose_data)
        pose_skeleton = normalization_transform(pose_skeleton)

        result = {
            'img': person_image_tensor,
            'img_agnostic': agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_skeleton,
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
        }
        return result

    @staticmethod
    def gen_noise(shape):
        noise = np.zeros(shape, dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise

    def generate_result(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        data = self.prepare_data(person_image, cloth_image, pose_data, person_segmentation, cloth_mask)
        img_agnostic = data['img_agnostic'][None, :]
        parse_agnostic = data['parse_agnostic'][None, :]
        pose = data['pose'][None, :]
        c = data['cloth'][None, :]
        cm = data['cloth_mask'][None, :]

        up = nn.Upsample(size=(self.fine_height, self.fine_width), mode='bilinear')
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
        pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
        c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
        cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
        seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, self.gen_noise(cm_down.size())), dim=1)

        parse_pred_down = self.seg_model(seg_input)
        parse_pred = gauss(up(parse_pred_down))
        parse_pred = parse_pred.argmax(dim=1)[:, None]

        parse_old = torch.zeros(parse_pred.size(0), 13, self.fine_height, self.fine_width, dtype=torch.float)
        parse_old.scatter_(1, parse_pred, 1.0)

        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        parse = torch.zeros(parse_pred.size(0), 7, self.fine_height, self.fine_width, dtype=torch.float)
        for j in range(len(labels)):
            for label in labels[j][1]:
                parse[:, j] += parse_old[:, label]

        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = self.gmm_model(gmm_input, c_gmm)
        warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

        misalign_mask = parse[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        # print(img_agnostic.size())
        # print(pose.size())
        # print(warped_c.size())
        # print(parse.size())
        # print(parse_div.size())
        # print(misalign_mask.size())

        output = self.alias_model(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

        output = (output.clone() + 1) * 0.5 * 255
        output = output.cpu().clamp(0, 255)

        try:
            output_array = output.numpy().astype('uint8')
        except:
            output_array = output.detach().numpy().astype('uint8')

        output_array = output_array.squeeze(0)
        output_array = output_array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(output_array)

        buffered = BytesIO()
        im.convert('RGB').save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("utf-8")
