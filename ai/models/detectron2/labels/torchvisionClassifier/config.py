'''
    2021 Benjamin Kellenberger
'''

from detectron2.config import CfgNode as CN

def add_torchvision_classifier_config(cfg):
    cfg.MODEL.META_ARCHITECTURE = 'TorchvisionClassifier'
    cfg.MODEL.TVCLASSIFIER = CN(new_allowed=True)
    cfg.MODEL.TVCLASSIFIER.NUM_CLASSES = 1000
    cfg.MODEL.TVCLASSIFIER.FLAVOR = "resnet50"
    cfg.MODEL.TVCLASSIFIER.PRETRAINED = True
    cfg.MODEL.PIXEL_MEAN: (0.485, 0.456, 0.406)
    cfg.MODEL.PIXEL_STD: (0.229, 0.224, 0.225)