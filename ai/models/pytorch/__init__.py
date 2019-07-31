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

# classification
from .functional.transforms import classification as ClassificationTransforms

# detection
from .functional.transforms import detection as DetectionTransforms



''' Utility functions '''
import torch
from util.helpers import get_class_executable


def parse_transforms(options):
    '''
        Recursively iterates through the options and initializes transform
        functions based on the given class executable name and kwargs.
    '''
    def _get_transform_executable(className):
        #TODO: dirty hack to be able to import custom transform functions...
        tokens = className.split('.')
        if tokens[-2] == 'classification':
            return getattr(ClassificationTransforms, tokens[-1])
        elif tokens[-2] == 'detection':
            return getattr(DetectionTransforms, tokens[-1])
        else:
            return get_class_executable(className)


    if isinstance(options, dict) and 'class' in options:
        tr_class = _get_transform_executable(options['class'])
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