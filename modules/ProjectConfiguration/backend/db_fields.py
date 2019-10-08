'''
    (Re-) Definition of annotation and prediction fields,
    along with their SQL setup instructions.

    2019 Benjamin Kellenberger
'''
#TODO: maybe merge with modules/LabelUI/backend/annotation_sql_tokens.py
#TODO: add "NOT NULL" option for label

from enum import Enum

class Fields_prediction(Enum):
    labels = ['label UUID']
    points = ['x REAL', 'y REAL', 'label UUID']
    boundingBoxes = ['x REAL', 'y REAL', 'width REAL', 'height REAL', 'label UUID']
    segmentationMasks = ['segmentationMask VARCHAR']


class Fields_annotation(Enum):
    labels = ['label UUID']
    points = ['x REAL', 'y REAL', 'label UUID']
    boundingBoxes = ['x REAL', 'y REAL', 'width REAL', 'height REAL', 'label UUID']
    segmentationMasks = ['segmentationMask VARCHAR', 'width REAL', 'height REAL']