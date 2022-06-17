'''
    Parsers for annotation im-/export.

    2022 Benjamin Kellenberger
'''

from .cocoParser import COCOparser
from .yoloParser import YOLOparser
from .segmentationParser import SegmentationFileParser

__all__ = (
    COCOparser,
    YOLOparser,
    SegmentationFileParser
)


# organize by annotation type and annotation format
PARSERS = {
    'labels': {},
    'points': {},
    'boundingBoxes': {
        'mscoco': COCOparser,
        'yolo': YOLOparser
    },
    'polygons': {},
    'segmentationMasks': {
        'image_files': SegmentationFileParser
    }
}


def auto_detect_parser(fileDict, annotationType, folderPrefix):
    '''
        Receives a dict of files (original: actual file name) and tries all
        parsers for a given annotation type on it. Returns the parser that was
        able to parse the files, or else None if no suitable parser was found.
        TODO: implement parser priority system.
    '''
    for pkey in PARSERS[annotationType].keys():
        if PARSERS[annotationType][pkey].is_parseable(fileDict, folderPrefix):
            return pkey