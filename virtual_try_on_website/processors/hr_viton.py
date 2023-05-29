import base64
import os
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchgeometry as tgm
from PIL import ImageDraw, Image

from virtual_try_on_website import settings
from virtual_try_on_website.models.hr_viton import ConditionGenerator, SPADEGenerator, make_grid
from virtual_try_on_website.processors.densepose import DensePose_Processor
from virtual_try_on_website.processors.lip_jppnet import LIP_JPPNet


class HR_VITON_Processor:
    def __init__(self):
        self.tocg_model = self.__load_tocg_model()
        self.generator_model = self.__load_generator_model()
        self.densepose_processor = DensePose_Processor()
        self.fine_height = 1024
        self.fine_width = 768

    @staticmethod
    def __load_tocg_model():
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = 16  # parse_agnostic + densepose
        model = ConditionGenerator(input1_nc=input1_nc, input2_nc=input2_nc, output_nc=13, ngf=96, norm_layer=nn.BatchNorm2d)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/hr_viton/mtviton.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def __load_generator_model():
        model = SPADEGenerator(3+3+3)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/hr_viton/gen.pth")
        state_dict = torch.load(state_dict_path)
        new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
        new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
        model.load_state_dict(new_state_dict, strict=True)

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

        agnostic = person_image.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
            mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(person_image, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(person_image, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(person_image, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic

    @staticmethod
    def __remove_overlap(seg_out, warped_cm):
        warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1))\
            .sum(dim=1, keepdim=True) * warped_cm
        return warped_cm

    def prepare_data(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        semantic_nc = 13
        normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cloth_image = Image.fromarray(cloth_image)
        cloth_image = transforms.Resize(self.fine_width, interpolation=2)(cloth_image)
        cloth_image = normalization_transform(cloth_image)

        cloth_mask = Image.fromarray(cloth_mask)
        cloth_mask = transforms.Resize(self.fine_width, interpolation=0)(cloth_mask)
        cloth_mask_array = np.array(cloth_mask)
        cloth_mask_array = (cloth_mask_array >= 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask_array)
        cloth_mask.unsqueeze_(0)

        person_image = Image.fromarray(person_image)
        person_image_tensor = transforms.Resize(self.fine_width, interpolation=2)(person_image)
        person_image_tensor = normalization_transform(person_image_tensor)

        person_segmentation_tensor = transforms.Resize(self.fine_width, interpolation=0)(person_segmentation)
        parse = torch.from_numpy(np.array(person_segmentation_tensor)[None]).long()

        labels = {
            0:  ['background',  [0, 10]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

        parse_agnostic = LIP_JPPNet.get_parse_agnostic(person_segmentation, pose_data)
        parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        person_cloth_mask = new_parse_map[3:4]
        person_cloth = person_image_tensor * person_cloth_mask + (1 - person_cloth_mask)


        person_densepose = self.densepose_processor.generate_densepose(person_image)
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(person_densepose)
        densepose_map = normalization_transform(densepose_map)

        agnostic = self.get_agnostic(person_image, person_segmentation, pose_data)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = normalization_transform(agnostic)

        return {
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'parse_onehot': parse_onehot,
            'parse': new_parse_map,
            'person_cloth_mask': person_cloth_mask,
            'parse_cloth': person_cloth,
            'image': person_image_tensor,
            'agnostic': agnostic
        }

    def generate_result(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        data = self.prepare_data(person_image, cloth_image, pose_data, person_segmentation, cloth_mask)
        pre_clothes_mask = data['cloth_mask'][None, :]
        label = data['parse'][None, :]
        parse_agnostic = data['parse_agnostic'][None, :]
        agnostic = data['agnostic'][None, :]
        clothes = data['cloth'][None, :]
        densepose = data['densepose'][None, :]
        im = data['image'][None, :]
        tocg_model = self.__load_tocg_model()
        generator_model = self.__load_generator_model()
        gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

        input_label, input_parse_agnostic = label, parse_agnostic
        pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float))

        pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
        input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
        input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
        agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
        clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
        densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

        shape = pre_clothes_mask.shape

        input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
        input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg_model(input1, input2)

        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float))

        cloth_mask = torch.ones_like(fake_segmap)
        cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
        fake_segmap = fake_segmap * cloth_mask

        fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(self.fine_height, self.fine_width), mode='bilinear'))
        fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

        old_parse = torch.FloatTensor(fake_parse.size(0), 13, self.fine_height, self.fine_width).zero_()
        old_parse.scatter_(1, fake_parse, 1.0)
        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        parse = torch.FloatTensor(fake_parse.size(0), 7, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse[:, i] += old_parse[:, label]

        N, _, iH, iW = clothes.shape
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)

        grid = make_grid(N, iH, iW)
        warped_grid = grid + flow_norm
        warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
        warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')

        warped_clothmask = self.__remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
        warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1 - warped_clothmask)

        output = generator_model(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)

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
