'''
    Contains SQL fragments for retrieval and submission of data
    for each of the annotation types.
    Strings still have to be formatted according to database schema.

    2019 Benjamin Kellenberger
'''

from enum import Enum

class QueryStrings_prediction(Enum):
    labels = '''
        SELECT id as predID, image, labelclass AS predLabel, confidence, priority FROM {}.prediction
    '''
    points = '''
        SELECT id as predID, image, x AS predX, y AS predY, labelclass AS predLabel, confidence, priority FROM {}.prediction
    '''
    boundingBoxes = '''
        SELECT id as predID, image, x AS predX, y AS predY, width AS predWidth, height AS predHeight, labelclass AS predLabel, confidence, priority FROM {}.prediction
    '''
    segmentationMasks = '''
        SELECT id as predID, image, filename AS predSegMap FROM {}.prediction
    '''


class QueryStrings_annotation(Enum):
    labels = '''
        SELECT id as annoID, image, labelclass AS annoLabel, confidence FROM {}.annotation
    '''
    points = '''
        SELECT id as annoID, image, x AS annoX, y AS annoY, labelclass AS annoLabel, confidence FROM {}.annotation
    '''
    boundingBoxes = '''
        SELECT id as annoID, image, x AS annoX, y AS annoY, width AS annoWidth, height AS annoHeight, labelclass AS annoLabel, confidence FROM {}.annotation
    '''
    segmentationMasks = '''
        SELECT id as annoID, image, filename AS annoSegMap FROM {}.annotation
    '''