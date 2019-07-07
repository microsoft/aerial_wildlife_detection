'''
    Enum for all column names in the database tables.

    2019 Benjamin Kellenberger
'''

from enum import Enum

class FieldNames_prediction(Enum):
    labels = set(['label', 'confidence'])
    points = set(['label', 'confidence', 'x', 'y'])
    boundingBoxes = set(['label', 'confidence', 'x', 'y', 'width', 'height'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename

class FieldNames_annotation(Enum):
    labels = set(['label', 'unsure'])
    points = set(['label', 'x', 'y', 'unsure'])
    boundingBoxes = set(['label', 'x', 'y', 'width', 'height', 'unsure'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename