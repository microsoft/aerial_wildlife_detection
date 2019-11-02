'''
    Built-in Tensorflow (wrapper) classes other than trainers and models
    are registered here for convenience.
    TODO: replace all with dedicated files.

    2019 Colin Torney
'''



''' Datasets '''

# boundingBoxes
from .functional.datasets.bboxDataset import BoundingBoxesDataset



''' Transforms '''

# boundingBoxes
from .functional.transforms import boundingBoxes as BoundingBoxesTransforms



''' Utility functions '''
from util.helpers import get_class_executable


def parse_transforms(options):
    '''
        Recursively iterates through the options and initializes transform
        functions based on the given class executable name and kwargs.
    '''
    def _get_transform_executable(className):
        #TODO: dirty hack to be able to import custom transform functions...
        tokens = className.split('.')
        if tokens[-2] == 'boundingBoxes':
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
