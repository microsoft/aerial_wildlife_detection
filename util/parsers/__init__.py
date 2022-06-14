'''
    Parsers for annotation im-/export.

    2022 Benjamin Kellenberger
'''

from .cocoParser import COCOparser
from .yoloParser import YOLOparser

__all__ = (
    COCOparser,
    YOLOparser
)


# organize by annotation type and annotation format
PARSERS = {
    'boundingBoxes': {
        'mscoco': COCOparser,
        'yolo': YOLOparser
    }
}


def auto_detect_parser(fileList, annotationType):
    '''
        Receives a list of files and tries all parsers for a given annotation
        type on it. Returns the parser that was able to parse the files, or else
        None if no suitable parser was found.
        TODO: implement parser priority system.
    '''
    for pkey in PARSERS[annotationType].keys():
        if PARSERS[annotationType][pkey].is_parseable(fileList):
            return pkey