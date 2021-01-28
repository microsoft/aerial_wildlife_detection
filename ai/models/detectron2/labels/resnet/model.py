'''
    2021 Benjamin Kellenberger
'''

import importlib
from typing import Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList

__all__ = ["GeneralizedResNet"]




@META_ARCH_REGISTRY.register()
class GeneralizedResNet(nn.Module):

    @configurable
    def __init__(self, *, model, pixel_mean, pixel_std):
        super().__init__()

        self.model = model
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    
    @classmethod
    def build_model(cls, cfg):
        resnetDef = importlib.import_module('torchvision.models.resnet')
        modelClass = getattr(resnetDef, cfg.MODEL.RESNET.FLAVOR)
        model = modelClass(pretrained=cfg.MODEL.RESNET.PRETRAINED)
        if cfg.MODEL.RESNET.NUM_CLASSES != 1000:
            # non-ImageNet model; adapt after initialization
            model.fc.weight = nn.Parameter(model.fc.weight[:cfg.MODEL.RESNET.NUM_CLASSES,...])
            model.fc.bias = nn.Parameter(model.fc.bias[:cfg.MODEL.RESNET.NUM_CLASSES])
        return model


    @classmethod
    def from_config(cls, cfg):
        model = cls.build_model(cfg)
        return {
            'model': model,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD
        }


    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        images = self.preprocess_image(batched_inputs)
        pred_logits = self.model(images.tensor)
        
        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "gt_label" in batched_inputs[0], "Ground truth label is missing in training!"
            gt_labels = torch.tensor([x["gt_label"] for x in batched_inputs]).to(self.device)
            loss = F.cross_entropy(pred_logits, gt_labels)
            return {
                'loss': loss
            }
        else:
            pred_logits = F.softmax(pred_logits, dim=1)
            pred_conf, pred_label = torch.max(pred_logits, dim=1)
            results = []
            for b in range(len(batched_inputs)):
                results.append({
                    'pred_label': pred_label[b],
                    'pred_conf': pred_conf[b],
                    'pred_logits': pred_logits[b,...]
                })
            return results


    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        '''
            Normalize and batch the input images.
        '''
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [x.float().div(255) for x in images if x.dtype != torch.float]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)
        return images