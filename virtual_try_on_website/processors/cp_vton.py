import base64
import os
from io import BytesIO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from virtual_try_on_website import settings
from virtual_try_on_website.models.cp_viton import CP_VTON_GMM, CP_VTON_TOM
from virtual_try_on_website.processors.lip_jppnet import LIP_JPPNet


class CP_VTON_Processor:
    def __init__(self):
        self.gmm_model = self.__load_gmm_model()
        self.tom_model = self.__load_tom_model()
        self.normalization_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.radius = 5

    @staticmethod
    def __load_gmm_model():
        model = CP_VTON_GMM()
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/cp_vton/gmm_final.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @staticmethod
    def __load_tom_model():
        model = CP_VTON_TOM(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/cp_vton/tom_final.pth")
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @classmethod
    def __generate_grid(cls, width, height, a=14, w=3):
        image = Image.new(mode='L', size=(width, height), color=0)
        draw = ImageDraw.Draw(image)

        for x in range(0, image.width, a + w):
            line = ((x, 0), (x, height))
            draw.line(line, fill=255, width=3)

        for y in range(0, image.height, a + w):
            line = ((0, y), (width, y))
            draw.line(line, fill=255, width=3)

        return image

    def __prepare_gmm_data(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        image_width = self.gmm_model.width
        image_height = self.gmm_model.height

        person_tensor = self.normalization_transform(person_image)
        cloth_tensor = self.normalization_transform(cloth_image)

        cloth_mask = (cloth_mask >= 128).astype(np.float32)
        cloth_mask_tensor = torch.from_numpy(cloth_mask).unsqueeze(0)

        grid = self.__generate_grid(image_width, image_height)
        grid_tensor = self.normalization_transform(grid)

        person_segmentation = np.array(person_segmentation)

        shape_mask, head_mask, cloth_parse_mask, body_mask = LIP_JPPNet.get_mask_arrays(person_segmentation)
        shape_mask = Image.fromarray((shape_mask * 255).astype(np.uint8))
        shape_mask = shape_mask.resize((image_width//16, image_height//16), Image.BILINEAR)
        shape_mask = shape_mask.resize((image_width, image_height), Image.BILINEAR)
        feature_shape_tensor = self.normalization_transform(shape_mask)

        head_mask_tensor = torch.from_numpy(head_mask)
        feature_head_tensor = person_tensor * head_mask_tensor - (1 - head_mask_tensor)

        cloth_parse_mask_tensor = torch.from_numpy(cloth_parse_mask)
        cloth_segmentation_tensor = person_tensor * cloth_parse_mask_tensor + (1 - cloth_parse_mask_tensor)

        body_mask_tensor = torch.from_numpy(body_mask).unsqueeze(0)

        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))
        point_num = pose_data.shape[0]
        feature_pose_tensor = torch.zeros(point_num, image_height, image_width)

        pose_representation = Image.new('L', (image_width, image_height))
        pose_draw = ImageDraw.Draw(pose_representation)
        for i in range(point_num):
            heatmap = Image.new('L', (image_width, image_height))
            draw = ImageDraw.Draw(heatmap)
            point_x = pose_data[i, 0]
            point_y = pose_data[i, 1]
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - self.radius, point_y - self.radius, point_x + self.radius, point_y + self.radius), 'white', 'white')
                pose_draw.rectangle((point_x - self.radius, point_y - self.radius, point_x + self.radius, point_y + self.radius), 'white', 'white')
            heatmap = self.normalization_transform(heatmap)
            feature_pose_tensor[i] = heatmap[0]

        pose_tensor = self.normalization_transform(pose_representation)

        cloth_agnostic_tensor = torch.cat([feature_shape_tensor, feature_head_tensor, feature_pose_tensor], 0)

        return {
            'person': person_tensor,
            'cloth': cloth_tensor,
            'cloth_mask': cloth_mask_tensor,
            'feature': cloth_agnostic_tensor,
            'pose': pose_tensor,
            'head': feature_head_tensor,
            'shape': feature_shape_tensor,
            'cloth_parse': cloth_segmentation_tensor,
            'body_mask': body_mask_tensor,
            'grid': grid_tensor
        }

    def __prepare_tom_data(self, warped_cloth, warped_cloth_mask):
        warped_cloth_tensor = self.normalization_transform(warped_cloth)

        warped_cloth_mask_tensor = (np.array(warped_cloth_mask) >= 128).astype(np.float32)
        warped_cloth_mask_tensor = torch.from_numpy(warped_cloth_mask_tensor).unsqueeze(0)

        return warped_cloth_tensor, warped_cloth_mask_tensor

    @torch.no_grad()
    def run_gmm_model(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        data = self.__prepare_gmm_data(person_image, cloth_image, pose_data, person_segmentation, cloth_mask)
        cloth = data['cloth'][None, :]
        cloth_mask = data['cloth_mask'][None, :]
        person = data['person'][None, :]
        body_mask = data['body_mask'][None, :]
        grid = data['grid'][None, :]

        result_grid, _ = self.gmm_model(data['feature'][None, :], cloth)
        # result_grid = torch.squeeze(result_grid)

        warped_cloth = F.grid_sample(cloth, result_grid, padding_mode='border')
        warped_cloth_mask = F.grid_sample(cloth_mask, result_grid, padding_mode='zeros')
        warped_cloth_mask = warped_cloth_mask * 2 - 1
        warped_grid = F.grid_sample(grid, result_grid, padding_mode='zeros')
        warped_person = body_mask * person + (1 - body_mask) * warped_cloth
        ground_truth = body_mask * person + (1 - body_mask) * data['cloth_parse']

        data['warped_cloth'] = warped_cloth
        data['warped_cloth_mask'] = warped_cloth_mask
        data['warped_grid'] = warped_grid

        return data

    def tensor_for_board(self, img_tensor):
        # map into [0,1]
        tensor = (img_tensor.clone() + 1) * 0.5
        tensor.cpu().clamp(0, 1)

        if tensor.size(1) == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        return tensor

    def tensor_list_for_board(self, img_tensors_list):
        grid_h = len(img_tensors_list)
        grid_w = max(len(img_tensors) for img_tensors in img_tensors_list)

        batch_size, channel, height, width = self.tensor_for_board(img_tensors_list[0][0]).size()
        canvas_h = grid_h * height
        canvas_w = grid_w * width
        canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
        for i, img_tensors in enumerate(img_tensors_list):
            for j, img_tensor in enumerate(img_tensors):
                offset_h = i * height
                offset_w = j * width
                tensor = self.tensor_for_board(img_tensor)
                canvas[:, :, offset_h: offset_h + height, offset_w: offset_w + width].copy_(tensor)
        return canvas

    def tensor_for_image(self, img_tensor):
        tensor = img_tensor.clone() * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        return array

    @torch.no_grad()
    def generate_result(self, person_image, cloth_image, pose_data, person_segmentation, cloth_mask):
        data = self.run_gmm_model(person_image, cloth_image, pose_data, person_segmentation, cloth_mask)

        warped_cloth = data['warped_cloth']
        warped_cloth_mask = data['warped_cloth_mask']
        # warped_cloth, warped_cloth_mask = self.__prepare_tom_data(warped_cloth, warped_cloth_mask)

        result = self.tom_model(torch.cat([data['feature'][None, :], warped_cloth], 1))
        rendered_person, composition_mask = torch.split(result, 3, 1)

        rendered_person = torch.tanh(rendered_person)
        composition_mask = torch.sigmoid(composition_mask)

        tryon_result_tensor = warped_cloth * composition_mask + rendered_person * (1 - composition_mask)
        tryon_result_tensor = self.tensor_for_board(tryon_result_tensor)
        tryon_result_array = self.tensor_for_image(tryon_result_tensor)
        tryon_result_array = tryon_result_array.swapaxes(0, 1).swapaxes(1, 2)
        tryon_result_image = Image.fromarray(tryon_result_array)

        #DEBUGGING
        # tryon_result_array = tryon_result_tensor.detach().numpy().astype('uint8')
        # if tryon_result_array.shape[0] == 3:
        #     tryon_result_array = tryon_result_array.swapaxes(0, 1).swapaxes(1, 2)

        # visuals = [[data['head'][None,:], data['shape'][None,:], data['pose'][None,:]],
        #            [warped_cloth, warped_cloth_mask * 2 - 1, composition_mask * 2 - 1],
        #            [rendered_person, tryon_result_tensor, data["person"]]]
        #
        # img_tensors = self.tensor_list_for_board(visuals)
        # for img_tensor, img_name in zip(img_tensors, "000254_1.jpg"):
        #     array = self.tensor_for_image(img_tensor)
        #     img = Image.fromarray(array)
        #     img.show()

        # output_image = Image.fromarray(tryon_result_array)

        output_image = tryon_result_image
        buffered = BytesIO()
        output_image.convert('RGB').save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("utf-8")
