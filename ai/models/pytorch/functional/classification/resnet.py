'''
    ResNet model for image classification.

    2019 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
from torchvision.models import resnet


class ResNet(nn.Module):

    def __init__(self, labelclassMap, featureExtractor='resnet50', pretrained=True):
        super(ResNet, self).__init__()

        self.labelclassMap = labelclassMap
        self.featureExtractor = featureExtractor
        self.pretrained = pretrained

        # create actual model
        if isinstance(featureExtractor, str):
            featureExtractor = getattr(resnet, featureExtractor)
        self.fe = featureExtractor(pretrained)

        #TODO: remove last layers, etc.