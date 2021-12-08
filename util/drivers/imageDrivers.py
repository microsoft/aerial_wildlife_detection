'''
    Utilities to load images of various formats.

    2021 Benjamin Kellenberger
'''

import mimetypes
mimetypes.init()

from io import BytesIO
import numpy as np


def bytea_to_bytesio(bytea):
    '''
        Returns a BytesIO wrapper around a given byte array (or the object
        itself if it already is a BytesIO instance).
        TODO: double definition, also in __init__ file...
    '''
    if isinstance(bytea, BytesIO):
        bytea.seek(0)
        return bytea
    else:
        bytesIO = BytesIO(bytea)
        bytesIO.seek(0)
        return bytesIO



class AbstractImageDriver:
    '''
        Abstract base class for image drivers. Convention: all drivers must
        return a NumPy ndarray of size (BxWxH), even if B=1. The number type or
        pixel values must not be changed from the original.
    '''

    NAME = 'abstract'
    PRIORITY = -1

    SUPPORTED_MEDIA_TYPES = ('image')

    SUPPORTED_EXTENSIONS = ()
    MULTIBAND_SUPPORTED = False

    @classmethod
    def init_is_available(cls):
        return False

    @classmethod
    def load_from_disk(cls, filePath):
        raise NotImplementedError('Not implemented for abstract base class.')
    
    @classmethod
    def load_from_bytes(cls, bytea):
        raise NotImplementedError('Not implemented for abstract base class.')
    
    @classmethod
    def save_to_disk(cls, array, filePath, **kwargs):
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def get_supported_extensions(cls):
        return cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_supported_mime_types(cls):
        return [mimetypes.types_map[s] for s in cls.SUPPORTED_EXTENSIONS if s in mimetypes.types_map]



class PILImageDriver(AbstractImageDriver):
    '''
        Uses the Python Image Library to load images. Fallback for when others
        (GDALImageDriver, etc.) don't work, as this one is limited to RGB data.
    '''

    NAME = 'PIL'
    PRIORITY = 10

    SUPPORTED_EXTENSIONS = (
        '.bmp',
        '.gif',
        '.icns',
        '.ico',
        '.jpg', '.jpeg',
        '.jp2', '.j2k',
        '.tif', '.tiff',
        '.webp'
    )

    @classmethod
    def init_is_available(cls):
        from PIL import Image
        cls.loader = Image
        return True

    @classmethod
    def load_from_disk(cls, filePath):
        arr = np.array(cls.loader.open(filePath))
        if arr.ndim == 2:
            return arr[np.newaxis,...]
        else:
            return arr.transpose((2,0,1))
    
    @classmethod
    def load_from_bytes(cls, bytea):
        arr = np.array(cls.loader.open(bytea_to_bytesio(bytea)))
        if arr.ndim == 2:
            return arr[np.newaxis,...]
        else:
            return arr.transpose((2,0,1))
    
    @classmethod
    def save_to_disk(cls, array, filePath, **kwargs):
        img = cls.loader.fromarray(array)
        img.save(filePath)



class GDALImageDriver(AbstractImageDriver):
    '''
        Uses GDAL via the rasterio bindings to load images.
    '''

    NAME = 'GDAL'
    PRIORITY = 20

    #TODO: list currently incomplete; see https://gdal.org/drivers/raster/index.html
    SUPPORTED_EXTENSIONS = (
        '.bmp',
        '.gif',
        '.heic',
        '.img',
        '.jpg', '.jpeg',
        '.jp2', '.j2k',
        '.nc',
        '.pdf',
        '.png',
        '.tif', '.tiff',
        '.webp'
    )

    @classmethod
    def init_is_available(cls):
        import rasterio
        cls.driver = rasterio
        from rasterio.io import MemoryFile
        cls.memfile = MemoryFile
        return True
    
    @classmethod
    def load_from_disk(cls, filePath):
        with cls.driver.open(filePath, 'r') as f:
            raster = f.read()
        return raster
    
    @classmethod
    def load_from_bytes(cls, bytea):
        with cls.memfile(bytea) as memfile:
            with memfile.open() as f:
                raster = f.read()
        return raster

    @classmethod
    def save_to_disk(cls, array, filePath, **kwargs):
        if isinstance(kwargs, dict):
            out_meta = kwargs
        else:
            out_meta = {}
        if 'width' not in out_meta:
            out_meta['width'] = array.shape[2]
        if 'height' not in out_meta:
            out_meta['height'] = array.shape[1]
        if 'count' not in out_meta:
            out_meta['count'] = array.shape[0]
        if 'dtype' not in out_meta:
            out_meta['dtype'] = str(array.dtype)
        with cls.driver.open(filePath, 'w', **out_meta) as dest_img:
            dest_img.write(array)



class DICOMImageDriver(AbstractImageDriver):
    '''
        Driver for DICOM files using pyDICOM.
    '''

    NAME = 'DICOM'
    PRIORITY = 5

    SUPPORTED_EXTENSIONS = (
        '.dcm',
    )

    @classmethod
    def init_is_available(cls):
        from pydicom import dcmread
        cls.loader = dcmread
        return True
    
    @classmethod
    def load_from_disk(cls, filePath):
        data = cls.loader(filePath)
        raster = data.pixel_array
        if raster.ndim == 2:
            raster = raster[np.newaxis,...]
        return raster
    
    @classmethod
    def load_from_bytes(cls, bytea):
        bytesIO = bytea_to_bytesio(bytea)
        data = cls.loader(bytesIO)
        raster = data.pixel_array
        if raster.ndim == 2:
            raster = raster[np.newaxis,...]
        return raster
    
    @classmethod
    def save_to_disk(cls, array, filePath, **kwargs):
        raise NotImplementedError('TODO: not yet implemented for DICOM parser.')


if __name__ == '__main__':
    drivers = (
        PILImageDriver,
        GDALImageDriver,
        DICOMImageDriver
    )

    for driver in drivers:
        dname = driver.NAME
        try:
            if not driver.init_is_available():
                raise Exception('driver not available')
            else:
                print(f'Driver "{dname}" initialized.')
        except Exception as e:
            print(f'Driver "{dname}" unavailable ("{str(e)}")')