'''
    Contains SQL fragments for retrieval and submission of data
    for each of the annotation types.

    2019 Benjamin Kellenberger
'''

from enum import Enum


class QueryStrings_prediction(Enum):
    labels = {
        'table': ['id', 'image', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predLabel', 'confidence', 'priority']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predLabel', 'confidence', 'priority']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'labelclass', 'confidence', 'priority'],
        'query': ['predID', 'image', 'predX', 'predY', 'predWidth', 'predHeight', 'predLabel', 'confidence', 'priority']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename', 'priority'],
        'query': ['predID', 'image', 'predSegMap', 'priority']
    }


class QueryStrings_annotation(Enum):
    labels = {
        'table': ['id', 'image', 'labelclass'],
        'query': ['annoID', 'image', 'annoLabel'],
        'submission': ['annotationID', None, 'label']
    }
    points = {
        'table': ['id', 'image', 'x', 'y', 'labelclass'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoLabel'],
        'submission': ['annotationID', None, 'x', 'y', 'label']
    }
    boundingBoxes = {
        'table': ['id', 'image', 'x', 'y', 'width', 'height', 'labelclass'],
        'query': ['annoID', 'image', 'annoX', 'annoY', 'annoWidth', 'annoHeight', 'annoLabel'],
        'submission': ['annotationID', None, 'x', 'y', 'width', 'height', 'label']
    }
    segmentationMasks = {
        'table': ['id', 'image', 'filename'],
        'query': ['annoID', 'image', 'annoSegMap'],
        'submission': ['annotationID', None, 'segmentationMap']
    }


def getQueryString(enum):
    return ', '.join(['{} AS {}'.format(enum['table'][x],enum['query'][x]) for x in range(len(enum['table']))])

def getTableNamesString(enum):
    return ', '.join([enum['table'][x] for x in range(len(enum['table']))])

def getOnConflictString(enum):
    return ', '.join(['{} = EXCLUDED.{}'.format(enum['table'][x],enum['table'][x]) for x in range(len(enum['table']))])




def parseAnnotation(annotation):
    '''
        Receives an annotation object as submitted by the labeling UI.
        Returns a tuple of values as well as a tuple of column names.
    '''
    values = []
    colnames = []

    # ID
    values.append(annotation['annotationID'])
    colnames.append('id')
    
    # geometry
    if 'geometry' in annotation:
        geom = annotation['geometry']
        values.extend(geom['coordinates'])
        type = geom['type']
        if type == 'point':
            colnames.extend(['x', 'y'])

        if type == 'rectangle':
            colnames.extend(['x', 'y', 'width', 'height'])

    # label
    if 'label' in annotation:
        values.append(annotation['label'])
        colnames.append('labelclass')

    return tuple(values), tuple(colnames)