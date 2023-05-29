import base64
import os
from io import BytesIO
from PIL import Image
from skimage import io, transform
import torch
from torch.autograd import Variable
from virtual_try_on_website import settings
import numpy as np

from virtual_try_on_website.models.u2netp import U2NETP


class U2NETP_Processor:
    def __init__(self):
        self.model = self.__load_model()

    class RescaleT(object):
        def __init__(self, output_size):
            self.output_size = output_size

        def __call__(self, sample):
            image, label = sample['image'], sample['label']

            img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
            lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                                   preserve_range=True)

            return {'image': img, 'label': lbl}

    @classmethod
    def __load_model(cls):
        model = U2NETP(3, 1)
        state_dict_path = os.path.join(settings.STATICFILES_DIRS[0], "models/u2netp.pth")
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        return model

    @classmethod
    def __transform_image(cls, cloth_image):
        output_size = 320

        image = transform.resize(cloth_image, (output_size, output_size), mode='constant')
        image = image / np.max(image)
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpImg = tmpImg.transpose((2, 0, 1))

        return torch.from_numpy(tmpImg)

    def calculate_mask(self, cloth_image):
        input_shape = cloth_image.shape

        cloth_image = self.__transform_image(cloth_image)
        cloth_image = cloth_image.type(torch.FloatTensor)
        cloth_image = Variable(cloth_image)

        d1, d2, d3, d4, d5, d6, d7 = self.model(cloth_image[None, :])

        prediction = d1[:, 0, :, :]
        pred_max = torch.max(prediction)
        pred_min = torch.min(prediction)
        prediction = (prediction - pred_min) / (pred_max - pred_min)

        prediction = prediction.squeeze()
        prediction_np = prediction.cpu().data.numpy()
        prediction_np = np.where(prediction_np >= 0.5, 1, 0)

        cloth_mask = Image.fromarray(prediction_np * 255).convert('RGB')
        cloth_mask = cloth_mask.resize((input_shape[1], input_shape[0]), resample=Image.BILINEAR)

        buffered = BytesIO()
        cloth_mask.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("utf-8")
