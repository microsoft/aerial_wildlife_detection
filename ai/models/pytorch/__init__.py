'''
    Built-in PyTorch (wrapper) classes are registered here for convenience.

    2019 Benjamin Kellenberger
'''


''' Models '''

# classification
from .functional.classification.resnet import ResNet

# detection
from .functional._retinanet.model import RetinaNet



''' Datasets '''

# classification
from .functional.datasets.classificationDataset import ClassificationDataset

# detection
from .functional.datasets.bboxDataset import BoundingBoxDataset



''' Transforms '''

# detection
from .functional._util import bboxTransforms