'''
    Contains SQL fragments for retrieval and submission of data
    for each of the annotation types.

    2019 Benjamin Kellenberger
'''

#BIG TODO: MERGE WITH sql_string_builder.py

from enum import Enum
from uuid import UUID



class FieldNames_prediction(Enum):
    labels = set(['label', 'confidence'])
    points = set(['label', 'confidence', 'x', 'y'])
    boundingBoxes = set(['label', 'confidence', 'x', 'y', 'width', 'height'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename

class FieldNames_annotation(Enum):
    labels = set(['label'])
    points = set(['label', 'x', 'y'])
    boundingBoxes = set(['label', 'x', 'y', 'width', 'height'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename




class QueryStrings_prediction(Enum):
    labels = {
        'table': ['id', 'image', 'label', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predLabel', 'predConfidence', 'priority']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'label', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predLabel', 'predConfidence', 'priority']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'label', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predWidth', 'predHeight', 'predLabel', 'predConfidence', 'priority']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename', 'priority'],
        'query': ['predID', 'image', 'predSegMap', 'priority']
    }


class QueryStrings_annotation(Enum):
    labels = {
        'table': ['id', 'image', 'label', 'username', 'timeCreated', 'timeRequired'],
        'query': ['annoID', 'image', 'annoLabel', 'annoUsername', 'annoTimecreated', 'annoTimerequired'],
        # 'submission': ['annotationID', None, 'label', 'username', 'timeCreated', 'timeRequired']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'label', 'username', 'timeCreated', 'timeRequired'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoLabel', 'annoUsername', 'annoTimecreated', 'annoTimerequired'],
        # 'submission': ['annotationID', None, 'x', 'y', 'label', 'username', 'timeCreated', 'timeRequired']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'label', 'username', 'timeCreated', 'timeRequired'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoWidth', 'annoHeight', 'annoLabel', 'annoUsername', 'annoTimecreated', 'annoTimerequired'],
        # 'submission': ['annotationID', None, 'x', 'y', 'width', 'height', 'label', 'username', 'timeCreated', 'timeRequired']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename', 'username', 'timeCreated', 'timeRequired'],
        'query': ['annoID', 'image', 'annoSegMap', 'annoUsername', 'annoTimecreated', 'annoTimerequired'],
        # 'submission': ['annotationID', None, 'segmentationMap', 'username', 'timeCreated', 'timeRequired']
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
            # check if new annotation (non-UUID conformant ID)
            # or update for existing
            try:
                value = str(UUID(annotation['id']))
            except:
                # non-UUID conformant ID: skip 'id' field
                continue
        
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