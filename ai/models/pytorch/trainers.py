'''
    Shorthand imports for model trainers that provide an interface between the AIWorker
    and the actual model.

    2019 Benjamin Kellenberger
'''


# labels
from .labels.resnet import ResNet

# points
from .points.wsodPointModel import WSODPointModel

# boundingBoxes
from .boundingBoxes.retinanet import RetinaNet