'''
    2020-21 Benjamin Kellenberger
'''

# models are listed here for convenience
from .labels.resnet.resnet import ResNet

from .boundingBoxes.fasterrcnn.fasterrcnn import FasterRCNN

from .boundingBoxes.retinanet.retinanet import RetinaNet

from .segmentationMasks.deeplabv3plus.deeplabv3plus import DeepLabV3Plus