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

__all__ = ["TorchvisionClassifier"]




@META_ARCH_REGISTRY.register()
class TorchvisionClassifier(nn.Module):

    CLASS_LAYER_MAP = {
        'alexnet': 'classifier.-1',
        'densenet161': 'classifier',
        'mnasnet1_0': 'classifier.-1',
        'mobilenet_v2': 'classifier.-1',
        'resnet18': 'fc',
        'resnet34': 'fc',
        'resnet50': 'fc',
        'resnet101': 'fc',
        'resnet152': 'fc',
        'resnext50_32x4d': 'fc',
        'squeezenet1_0': 'classifier.1',
        'shufflenet_v2_x1_0': 'fc',
        'vgg16': 'classifier.-1',
        'wide_resnet50_2': 'fc',
        'wide_resnet101_2': 'fc',
        #TODO: the following models require scipy and are thus currently not supported
        # 'inception_v3': '',
        # 'googlenet': '',
    }

    @configurable
    def __init__(self, *, model, pixel_mean, pixel_std):
        super().__init__()

        self.model = model
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    
    @classmethod
    def get_classification_layer(cls, model, flavor):
        assert flavor in cls.CLASS_LAYER_MAP, f'Invalid model type specified ("{flavor}").'
        tokens = cls.CLASS_LAYER_MAP[flavor].split('.')
        if hasattr(model, 'model'):
            # one level up; prepend token
            tokens.insert(0, 'model')
        layer = model
        for t in tokens:
            try:
                layer = layer[int(t)]
            except:
                layer = getattr(layer, t)
        return layer

    
    @classmethod
    def set_classification_layer(cls, model, layer, flavor):
        assert flavor in cls.CLASS_LAYER_MAP, f'Invalid model type specified ("{flavor}").'
        tokens = cls.CLASS_LAYER_MAP[flavor].split('.')
        if hasattr(model, 'model'):
            # one level up; prepend token
            tokens.insert(0, 'model')
        item = model
        for idx, t in enumerate(tokens):
            if idx == len(tokens)-1:
                try:
                    item[int(t)] = layer
                except:
                    setattr(item, t, layer)
            else:
                try:
                    item = item[int(t)]
                except:
                    item = getattr(item, t)

    
    @classmethod
    def build_model(cls, cfg):
        modelDef = importlib.import_module('torchvision.models')
        modelClass = getattr(modelDef, cfg.MODEL.TVCLASSIFIER.FLAVOR)
        model = modelClass(pretrained=cfg.MODEL.TVCLASSIFIER.PRETRAINED)
        if cfg.MODEL.TVCLASSIFIER.NUM_CLASSES != 1000:
            # non-ImageNet model; adapt after initialization
            classLayer = cls.get_classification_layer(model, cfg.MODEL.TVCLASSIFIER.FLAVOR)
            classLayer.weight = nn.Parameter(classLayer.weight[:cfg.MODEL.TVCLASSIFIER.NUM_CLASSES,...])
            classLayer.bias = nn.Parameter(classLayer.bias[:cfg.MODEL.TVCLASSIFIER.NUM_CLASSES])
            if hasattr(classLayer, 'out_features'):
                classLayer.out_features = cfg.MODEL.TVCLASSIFIER.NUM_CLASSES
            cls.set_classification_layer(model, classLayer, cfg.MODEL.TVCLASSIFIER.FLAVOR)
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