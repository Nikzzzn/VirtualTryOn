import os
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from virtual_try_on_website import settings
from virtual_try_on_website.models.DensePose.densepose import add_densepose_config, DensePoseDataRelative
from virtual_try_on_website.models.DensePose.densepose.structures import DensePoseChartPredictorOutput, \
    DensePoseEmbeddingPredictorOutput
from virtual_try_on_website.models.DensePose.densepose.vis.base import MatrixVisualizer
from virtual_try_on_website.models.DensePose.densepose.vis.extractor import DensePoseResultExtractor, \
    DensePoseOutputsExtractor


class DensePose_Processor:
    def __init__(self):
        cfg = self.setup_config()
        self.predictor = DefaultPredictor(cfg)

    @torch.no_grad()
    def generate_densepose(self, person_image):
        outputs = self.predictor(person_image)["instances"]

        if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
            extractor = DensePoseResultExtractor()
        elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
            extractor = DensePoseOutputsExtractor()

        pred_densepose = extractor(outputs)[0]

        bbox_xywh = outputs.pred_boxes.tensor.clone()
        bbox_xywh[:, 2] -= bbox_xywh[:, 0]
        bbox_xywh[:, 3] -= bbox_xywh[:, 1]
        bbox_xywh = bbox_xywh[0].cpu().numpy()

        iuv_array = torch.cat((pred_densepose[0].labels[None].type(torch.float32), pred_densepose[0].uv * 255.0)).type(torch.uint8).cpu().numpy()

        matrix = iuv_array[0, :, :]
        segm = iuv_array[0, :, :]
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1

        mask_visualizer = MatrixVisualizer(inplace=False, cmap=cv2.COLORMAP_PARULA, val_scale=DensePoseDataRelative.N_PART_LABELS, alpha=1.0)

        res_img = mask_visualizer.visualize(person_image, mask, matrix, bbox_xywh)

        return res_img

    @staticmethod
    def setup_config():
        cfg = get_cfg()
        add_densepose_config(cfg)
        model_path = os.path.join(settings.STATICFILES_DIRS[0], "densepose", "model_final_162be9.pkl")
        cfg.merge_from_file("../models/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.freeze()

        return cfg