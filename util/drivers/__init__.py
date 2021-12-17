'''
    Utilities to load data (images, etc.) of various formats.

    2021 Benjamin Kellenberger
'''

from util.drivers.imageDrivers import (
    PILImageDriver,
    GDALImageDriver,
    DICOMImageDriver
)

from util.helpers import LogDecorator

import os
from functools import cmp_to_key
from io import BytesIO

import magic


DRIVERS = {
    'extension': {},
    'mime_type': {}
}


'''
    A - valid file extensions and MIME types for files that are fully supported
    by AIDE, both in the Web frontend as well as the backend through the image
    drivers.
'''
SUPPORTED_DATA_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.tif',
    '.tiff',
    '.bmp',
    '.ico',
    '.jfif',
    '.pjpeg',
    '.pjp',
    '.dcm'
)

SUPPORTED_DATA_MIME_TYPES = (
    'image/jpeg',
    'image/bmp',
    'image/x-windows-bmp',
    'image/gif',
    'image/tif',
    'image/tiff',
    'image/x-icon',
    'image/png',
    'application/dicom'
)

# data extensions and MIME types to use for conversion
DATA_EXTENSIONS_CONVERSION = {
    'image': '.tif'
}

DATA_MIME_TYPES_CONVERSION = {
    'image': 'image/tif'
}


'''
    B - valid file extensions and MIME types that are supported by the drivers,
    but not necessarily by AIDE's Web frontend. For such files, AIDE offers the
    option to convert them to a format that is versatile enough and fully
    supported (e.g., TIFF for images).
'''
VALID_IMAGE_EXTENSIONS = set()
VALID_IMAGE_MIME_TYPES = set()

# fallback image drivers - those that are to be tried in order if data extensions
# or MIME types are not conclusive
FALLBACK_DRIVERS = (
    GDALImageDriver, DICOMImageDriver
)

def init_drivers(verbose=False):
    if len(VALID_IMAGE_EXTENSIONS):
        return
    if verbose:
        print('Initializing data drivers...')
    for driver in (PILImageDriver, GDALImageDriver, DICOMImageDriver):
        if verbose:
            print(f'    {driver.NAME}'.ljust(LogDecorator.get_ljust_offset()), end='')
        try:
            if not driver.init_is_available():
                raise Exception('unknown error')
            
            for ext in driver.get_supported_extensions():
                if ext not in DRIVERS['extension']:
                    DRIVERS['extension'][ext] = set()
                DRIVERS['extension'][ext].add(driver)
            
            for mimetype in driver.get_supported_mime_types():
                if mimetype not in DRIVERS['mime_type']:
                    DRIVERS['mime_type'][mimetype] = set()
                DRIVERS['mime_type'][mimetype].add(driver)
            if verbose:
                LogDecorator.print_status('ok')
        except Exception as e:
            if verbose:
                LogDecorator.print_status('fail')
                print(f'        .> {str(e)}')
    
    # sort drivers according to priority
    def _sort_priority(a, b):
        if a.PRIORITY < b.PRIORITY:
            return 1
        elif a.PRIORITY == b.PRIORITY:
            return 0
        else:
            return -1
    
    for ext in DRIVERS['extension'].keys():
        DRIVERS['extension'][ext] = list(DRIVERS['extension'][ext])
        DRIVERS['extension'][ext].sort(key=cmp_to_key(_sort_priority))
        VALID_IMAGE_EXTENSIONS.add(ext)
    for mimetype in DRIVERS['mime_type'].keys():
        DRIVERS['mime_type'][mimetype] = list(DRIVERS['mime_type'][mimetype])
        DRIVERS['mime_type'][mimetype].sort(key=cmp_to_key(_sort_priority))
        VALID_IMAGE_MIME_TYPES.add(mimetype)


def get_drivers_by_mime_type(mimeType, omit_fallback=False):
    if not len(DRIVERS['mime_type']):
        init_drivers(False)
    try:
        return DRIVERS['mime_type'][mimeType]
    except:
        return FALLBACK_DRIVERS



def get_drivers_by_extension(ext, omit_fallback=False):
    if not len(DRIVERS['extension']):
        init_drivers(False)
    try:
        return DRIVERS['extension'][ext]
    except:
        return FALLBACK_DRIVERS



def bytea_to_bytesio(bytea):
    '''
        Returns a BytesIO wrapper around a given byte array (or the object
        itself if it already is a BytesIO instance).
    '''
    if isinstance(bytea, BytesIO):
        bytea.seek(0)
        return bytea
    else:
        bytesIO = BytesIO(bytea)
        bytesIO.seek(0)
        return bytesIO


def bytesio_to_bytea(bytesio):
    '''
        Returns a bytes array from a given BytesIO object (or the object itself
        if it already is a bytes array).
    '''
    if isinstance(bytesio, BytesIO):
        bytesio.seek(0)
        return bytesio.read()
    else:
        return bytesio



def load_from_disk(filePath, override_extension=None, return_driver=False):
    '''
        Tries to guess the driver from the file name's extension. Iterates
        through available drivers, sorted by priority, and tries to load the
        file. Throws an exception if:
        - no driver can be found for the given extension
        - none of the available drivers can load the file

        The file's extension may be overridden, if desired.
    '''
    ext = override_extension
    if isinstance(ext, str):
        ext = ext.lower()
        if not ext[0] == '.':
            ext = '.' + ext
    else:
        _, ext = os.path.splitext(filePath)
    
    drivers = get_drivers_by_extension(ext)
    for driver in drivers:
        try:
            result = [driver.load_from_disk(filePath)]
            if return_driver:
                result.append(driver)
            return tuple(result) if len(result) > 1 else result[0]
        except:
            pass
    # no driver was able to load the file
    raise Exception(f'None of the available drivers could load file "{filePath}".')



def load_from_bytes(bytea, return_mime_type=False, return_driver=False):
    '''
        Guesses the MIME type from the provided byte array using the libmagic
        library. Then, parses the byte array using an appropriate driver, if
        available.
    '''
    mimeType = magic.from_buffer(bytesio_to_bytea(bytea), mime=True).lower()      #TODO: workaround if libmagic fails?
    drivers = get_drivers_by_mime_type(mimeType)
    for driver in drivers:
        try:
            result = [driver.load_from_bytes(bytea)]
            if return_mime_type:
                result.append(mimeType)
            if return_driver:
                result.append(driver)
            return tuple(result) if len(result) > 1 else result[0]
        except:
            pass
    # no driver was able to load the file
    raise Exception(f'None of the available drivers could load bytes (guessed MIME type: "{mimeType}").')



def save_to_disk(array, filePath, **kwargs):
    '''
        Receives a NumPy ndarray and a file path and saves the image to disk.
        Guesses the driver from the file name extension.
    '''
    parent, fname = os.path.split(filePath)
    _, ext = os.path.splitext(fname)
    drivers = get_drivers_by_extension(ext)
    if len(parent):
        os.makedirs(parent, exist_ok=True)
    for driver in drivers:
        driver.save_to_disk(array, filePath, **kwargs)
        break



if __name__ == '__main__':
    init_drivers(True)