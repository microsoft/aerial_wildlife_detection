'''
    2020-22 Benjamin Kellenberger
'''

# models are listed here for convenience
from .labels.torchvisionClassifier.torchvisionClassifier import GeneralizedTorchvisionClassifier
from .labels.torchvisionClassifier.alexnet import AlexNet
from .labels.torchvisionClassifier.densenet import DenseNet161
from .labels.torchvisionClassifier.mnasnet import MnasNet
from .labels.torchvisionClassifier.mobilenet import MobileNetV2
from .labels.torchvisionClassifier.resnet import (ResNet18,
                                                ResNet34,
                                                ResNet50,
                                                ResNet101,
                                                ResNet152,
                                                WideResNet50,
                                                WideResNet101)
from .labels.torchvisionClassifier.resnext import ResNeXt50, ResNeXt101
from .labels.torchvisionClassifier.shufflenet import ShuffleNetV2
from .labels.torchvisionClassifier.squeezenet import SqueezeNet
from .labels.torchvisionClassifier.vgg import VGG16

from .boundingBoxes.fasterrcnn.fasterrcnn import FasterRCNN

from .boundingBoxes.retinanet.retinanet import RetinaNet

from .boundingBoxes.tridentnet.tridentnet import TridentNet

from .boundingBoxes.yolov5.yolo import YOLOv5

from .boundingBoxes.deepforest.deepforest import DeepForest

from .segmentationMasks.deeplabv3plus.deeplabv3plus import DeepLabV3Plus

from .segmentationMasks.unet.unet import Unet

from .instanceSegmentation.maskrcnn.maskrcnn import MaskRCNN