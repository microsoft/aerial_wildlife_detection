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



''' Utility functions '''
import torch
from util.helpers import get_class_executable

def parse_transforms(options):
    '''
        Recursively iterates through the options and initializes transform
        functions based on the given class executable name and kwargs.
    '''
    if isinstance(options, dict) and 'class' in options:
        tr_class = get_class_executable(options['class'])
        if 'kwargs' in options:
            for kw in options['kwargs']:
                options['kwargs'][kw] = parse_transforms(options['kwargs'][kw])
            tr_inst = tr_class(**options['kwargs'])
        else:
            tr_inst = tr_class()
        options = tr_inst
    elif isinstance(options, list):
        for o in range(len(options)):
            options[o] = parse_transforms(options[o])
    return options


def get_device(options):
    device = options['general']['device']
    if 'cuda' in device and not torch.cuda.is_available():
        device = 'cpu'
    return device