'''
    2020-21 Benjamin Kellenberger
'''

# models are listed here for convenience
from .labels.torchvisionClassifier.torchvisionClassifier import TorchvisionClassifier

from .boundingBoxes.fasterrcnn.fasterrcnn import FasterRCNN

from .boundingBoxes.retinanet.retinanet import RetinaNet

from .segmentationMasks.deeplabv3plus.deeplabv3plus import DeepLabV3Plus