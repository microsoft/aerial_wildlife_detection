'''
    Built-in PyTorch (wrapper) classes are registered here for convenience.

    2019 Benjamin Kellenberger
'''


''' Models '''

# abstract base
from .genericPyTorchModel import GenericPyTorchModel

# labels
from .functional.classification.resnet import ResNet

# points
# from .functional._wsodPoints.model import WSODPointModel      #TODO: implement

# boundingBoxes
from .functional._retinanet.model import RetinaNet



''' Datasets '''

# labels
from .functional.datasets.classificationDataset import LabelsDataset

# points
from .functional.datasets.pointsDataset import PointsDataset

# boundingBoxes
from .functional.datasets.bboxDataset import BoundingBoxesDataset



''' Transforms '''

# labels
from .functional.transforms import labels as LabelsTransforms

# points
# from .functional.transforms imoprt points as PointsTransforms     #TODO: implement

# boundingBoxes
from .functional.transforms import boundingBoxes as BoundingBoxesTransforms



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
        if tokens[-2] == 'labels':
            return getattr(LabelsTransforms, tokens[-1])
        elif tokens[-2] == 'points':
            # return getattr(PointsTransforms, tokens[-1])  #TODO: implement
            return None
        elif tokens[-2] == 'boundingBoxes':
            return getattr(BoundingBoxesTransforms, tokens[-1])
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