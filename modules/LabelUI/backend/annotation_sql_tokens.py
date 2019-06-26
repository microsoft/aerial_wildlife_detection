'''
    Contains SQL fragments for retrieval and submission of data
    for each of the annotation types.

    2019 Benjamin Kellenberger
'''

from enum import Enum
from uuid import UUID


class QueryStrings_prediction(Enum):
    labels = {
        'table': ['id', 'image', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predLabel', 'predConfidence', 'priority']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predLabel', 'predConfidence', 'priority']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predWidth', 'predHeight', 'predLabel', 'predConfidence', 'priority']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename', 'priority'],
        'query': ['predID', 'image', 'predSegMap', 'priority']
    }


class QueryStrings_annotation(Enum):
    labels = {
        'table': ['id', 'image', 'labelclass', 'username'],
        'query': ['annoID', 'image', 'annoLabel', 'annoUsername'],
        'submission': ['annotationID', None, 'label', 'username']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'labelclass', 'username'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoLabel', 'annoUsername'],
        'submission': ['annotationID', None, 'x', 'y', 'label', 'username']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'labelclass', 'username'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoWidth', 'annoHeight', 'annoLabel', 'annoUsername'],
        'submission': ['annotationID', None, 'x', 'y', 'width', 'height', 'label', 'username']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename', 'username'],
        'query': ['annoID', 'image', 'annoSegMap', 'annoUsername'],
        'submission': ['annotationID', None, 'segmentationMap', 'username']
    }


def getQueryString(enum):
    return ', '.join(['{} AS {}'.format(enum['table'][x],enum['query'][x]) for x in range(len(enum['table']))])

def getTableNames(enum):
    return [enum['table'][x] for x in range(len(enum['table']))]

def getOnConflictString(enum):
    return ', '.join(['{} = EXCLUDED.{}'.format(enum['table'][x],enum['table'][x]) for x in range(len(enum['table'])) if enum['table'][x] != 'id'])




def parseAnnotation(annotation):
    '''
        Receives an annotation object as submitted by the labeling UI.
        Returns a dictionary of column names and values.
    '''
    valuesDict = {}

    for key in annotation.keys():
        if key == 'id':
            # replace annotation ID with 'DEFAULT' keyword if no UUID
            # (this is the case if the annotation is not yet in the DB)
            try:
                value = str(UUID(annotation['id']))
            except:
                value = 'DEFAULT'
        
        elif key == 'geometry':
            # iterate through geometry tokens
            for subKey in annotation['geometry'].keys():
                valuesDict[subKey] = annotation['geometry'][subKey]     #TODO: typecast required?
            continue
        
        else:
            # generic parameter
            value = annotation[key]
    
        valuesDict[key] = value

    return valuesDict