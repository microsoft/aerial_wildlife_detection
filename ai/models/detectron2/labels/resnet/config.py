'''
    2021 Benjamin Kellenberger
'''

from detectron2.config import CfgNode as CN

def add_resnet_config(cfg):
    cfg.MODEL.META_ARCHITECTURE = 'ResNet'
    cfg.MODEL.RESNET = CN(new_allowed=True)
    cfg.MODEL.RESNET.NUM_CLASSES = 1000
    cfg.MODEL.RESNET.FLAVOR = "resnet50"
    cfg.MODEL.RESNET.PRETRAINED = True