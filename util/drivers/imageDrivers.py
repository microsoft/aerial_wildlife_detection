'''
    Utilities to load images of various formats.

    2021-23 Benjamin Kellenberger
'''

from io import BytesIO
import mimetypes
import numpy as np

mimetypes.init()


DEFAULT_IMAGE_METADATA = {
    'dtpye': '',
    'width': 0,
    'height': 0,
    'count': 0,
    'descriptions': (),
    'bounds': None,
    'transform': None,
    'crs': None
}


def bytea_to_bytesio(bytea: object) -> BytesIO:
    '''
        Returns a BytesIO wrapper around a given byte array (or the object itself if it already is a
        BytesIO instance). TODO: double definition, also in __init__ file...
    '''
    if isinstance(bytea, BytesIO):
        bytea.seek(0)
        return bytea
    else:
        bytes_io = BytesIO(bytea)
        bytes_io.seek(0)
        return bytes_io



def bytesio_to_bytea(bytes_io: object) -> bytes:
    '''
        Returns a byte array from a BytesIO wrapper.
    '''
    if isinstance(bytes_io, BytesIO):
        bytes_io.seek(0)
        return bytes_io.getvalue()
    return bytes_io


def band_selection(img: np.array, bands: tuple=(1,2,3)) -> np.array:
    '''
        Band selection. Band indices start at 0.
    '''
    if bands is None:
        return img
    return img.take(bands, axis=0)


def normalize_image(img: np.array, band_axis: int=0, color_range: object=255) -> np.array:
    '''
        Receives an image in np.array format and normalizes it into a [0, 255] uint8 image.
        Parameter "band_axis" determines in which axis the image bands can be found. Parameter
        "color_range" defines the maximum obtainable integer value per band. Default is 255 (full
        uint8 range), but lower values may be specified to e.g. perform a crude quantization of the
        image color space.
    '''
    if img.ndim == 2:
        img = img[np.newaxis,...]
        band_axis = 0
    if not isinstance(color_range, int) and not isinstance(color_range, float):
        color_range = 255
    else:
        color_range = int(min(255, max(0, color_range)))
    permuted = False
    if band_axis != 0:
        permuted = True
        if band_axis < 0:
            band_axis = img.ndim + band_axis
        band_order = list(range(img.ndim))
        band_order.remove(band_axis)
        band_order.insert(0,band_axis)
        img = np.transpose(img, (band_order))
    size = img.shape
    img = img.reshape((size[0], -1)).astype(np.float32)
    mins = np.min(img, 1)[:,np.newaxis]
    maxs = np.max(img, 1)[:,np.newaxis]
    img = (img - mins)/(maxs - mins)
    img = color_range * img.reshape(size)
    if permuted:
        # permute back
        img = np.transpose(img, np.argsort(band_order))
    img = img.astype(np.uint8)
    return img



class AbstractImageDriver:
    '''
        Abstract base class for image drivers. Convention: all drivers must return a NumPy ndarray
        of size (BxWxH), even if B=1. The number type or pixel values must not be changed from the
        original.
    '''

    NAME = 'abstract'
    PRIORITY = -1

    SUPPORTED_MEDIA_TYPES = ('image')

    SUPPORTED_EXTENSIONS = ()
    MULTIBAND_SUPPORTED = False

    @classmethod
    def init_is_available(cls):
        '''
            Returns True if pre-flight checks have passed (e.g., dependencies are available).
        '''
        return False

    @classmethod
    def is_loadable(cls, obj):
        '''
            Receives an object of any form and returns True if it can be loaded with the current
            driver (else False).
        '''
        return False

    @classmethod
    def metadata(cls, obj, **kwargs):
        '''
            Returns metadata for a given object (either a str or bytes array), such as geospatial
            registration info.
        '''
        return DEFAULT_IMAGE_METADATA

    @classmethod
    def size(cls, obj):
        '''
            Returns the size as [count, height, width] for a given object, assuming it is parseable.
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def load(cls, obj, **kwargs):
        '''
            Loads an image from object (either a str or bytes array).
        '''
        if isinstance(obj, str):
            return cls.load_from_disk(obj, **kwargs)
        return cls.load_from_bytes(obj, **kwargs)

    @classmethod
    def load_from_disk(cls, file_path, **kwargs):
        '''
            Loads an image from disk from a given "file_path" (str).
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def load_from_bytes(cls, bytea, **kwargs):
        '''
            Loads an image from a given byte array "bytea" (bytes).
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def save_to_disk(cls, array, file_path, **kwargs):
        '''
            Saves a given array from memory to disk.
            Inputs:
            - "array": NumPy.array, array of size [count, height, width]
            - "file_path": str, path on file system to save image to.
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def save_to_bytes(cls, array, image_format, **kwargs):
        '''
            Saves a given array from memory to an in-memory image file.
            Inputs:
            - "array": NumPy.array, array of size [count, height, width]
            - "image_format": str, format of image (e.g., "tiff")
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def disk_to_bytes(cls, file_path, **kwargs):
        '''
            Loads an image from disk and saves it to an in-memory image. Useful to e.g. only load a
            spatial sub-portion of an image (windowed reading), specifiable via "kwargs".
        '''
        raise NotImplementedError('Not implemented for abstract base class.')

    @classmethod
    def get_supported_extensions(cls):
        '''
            Returns a list of file extensions this specific driver can parse.
        '''
        return cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_supported_mime_types(cls):
        '''
            Returns a list of MIME types (str) this specific driver can parse.
        '''
        return [mimetypes.types_map[s] for s in cls.SUPPORTED_EXTENSIONS
                if s in mimetypes.types_map]



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
    def _load(cls, obj, **kwargs):
        img = None
        if isinstance(obj, str):
            img = cls.loader.open(obj)
        else:
            img = cls.loader.open(bytea_to_bytesio(obj))
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            img = img.crop(*kwargs['window'])
        return img

    @classmethod
    def init_is_available(cls):
        from PIL import Image
        cls.loader = Image
        return True

    @classmethod
    def is_loadable(cls, obj):
        img = None
        try:
            img = cls._load(obj)
            img.verify()
            img.close()     #TODO: needed?
            return True
        except Exception:
            return False
        finally:
            if hasattr(img, 'close'):
                img.close()

    @classmethod
    def metadata(cls, obj, **kwargs):
        meta = DEFAULT_IMAGE_METADATA
        img = cls._load(obj, kwargs)
        meta.update({
            'dtype': 'uint8',       #TODO
            'count': img.layers,
            'width': img.width,
            'height': img.height
        })
        return meta

    @classmethod
    def size(cls, obj):
        img = cls._load(obj)
        size = [len(img.getbands()), img.height, img.width]
        img.close()
        return size

    @classmethod
    def load_from_disk(cls, file_path, **kwargs):
        img = cls.loader.open(file_path)
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            img = img.crop(*kwargs['window'])
        arr = np.array(img)
        if arr.ndim == 2:
            return arr[np.newaxis,...]
        return arr.transpose((2,0,1))

    @classmethod
    def load_from_bytes(cls, bytea, **kwargs):
        img = cls.loader.open(bytea_to_bytesio(bytea))
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            img = img.crop(*kwargs['window'])
        arr = np.array(img)
        if arr.ndim == 2:
            return arr[np.newaxis,...]
        return arr.transpose((2,0,1))

    @classmethod
    def save_to_disk(cls, array, file_path, **kwargs):
        img = cls.loader.fromarray(array)
        img.save(file_path)

    @classmethod
    def save_to_bytes(cls, array, image_format, **kwargs):
        bio = BytesIO()
        img = cls.loader.fromarray(array)
        img.save(bio, format=image_format)
        return bio

    @classmethod
    def disk_to_bytes(cls, file_path, **kwargs):
        img = cls.loader.open(file_path)
        if 'window' in kwargs:
            # crop
            img = img.crop(*kwargs['window'])
        bio = BytesIO()
        img.save(bio, format=img.format)
        return bio



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
        '.gff',
        '.gpkg',
        '.grd',
        '.heic',
        '.img',
        '.jpg', '.jpeg',
        '.jp2', '.j2k',
        '.nc',
        '.ntf',
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
        from rasterio.windows import Window
        cls.window = Window

        # filter warnings
        import warnings
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        from rasterio import logging
        log = logging.getLogger()
        log.setLevel(logging.ERROR)
        return True

    @classmethod
    def is_loadable(cls, obj):
        try:
            if isinstance(obj, str):
                with cls.driver.open(obj, 'r'):
                    valid = True    #TODO
            else:
                with cls.memfile(bytesio_to_bytea(obj)) as memfile:
                    with memfile.open():
                        valid = True
            return valid
        except Exception:
            return False

    @classmethod
    def metadata(cls, obj, **kwargs):
        '''
            Returns metadata for a given object (str, bytes array, rasterio.Dataset), such as
            geospatial registration info.
        '''
        meta = DEFAULT_IMAGE_METADATA
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            window = cls.window(*kwargs['window'])
        else:
            window = None
        if isinstance(obj, cls.driver.DatasetReader):
            f_raster = obj
        if isinstance(obj, str):
            f_raster = cls.driver.open(obj, 'r')
        else:
            with cls.memfile(bytesio_to_bytea(obj)) as memfile:
                f_raster = memfile.open()
        with f_raster:
            profile = f_raster.profile
            profile['descriptions'] = f_raster.descriptions
            if window is not None:
                profile['bounds'] = f_raster.window_bounds(window)
                profile['transform'] = f_raster.window_transform(window)
            else:
                profile['bounds'] = (
                    f_raster.bounds.left,
                    f_raster.bounds.bottom,
                    f_raster.bounds.right,
                    f_raster.bounds.top
                )
        meta.update(profile)
        return meta

    @classmethod
    def size(cls, obj):
        if isinstance(obj, str):
            with cls.driver.open(obj, 'r') as f_raster:
                size = [f_raster.count, f_raster.height, f_raster.width]
        else:
            with cls.memfile(bytesio_to_bytea(obj)) as memfile:
                with memfile.open() as f_raster:
                    size = [f_raster.count, f_raster.height, f_raster.width]
        return size

    @classmethod
    def load_from_disk(cls, file_path, **kwargs):
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            window = cls.window(*kwargs['window'])
        else:
            window = None
        with cls.driver.open(file_path, 'r') as f_raster:
            raster = f_raster.read(window=window, boundless=True)
            return_args = [raster]
            if kwargs.get('return_metadata', False):
                profile = cls.metadata(f_raster, window=window)
                return_args.append(profile)

            bands = kwargs.get('bands', None)
            if bands is not None:
                raster = band_selection(raster, bands)
                profile['count'] = len(bands)

            return tuple(return_args)

    @classmethod
    def load_from_bytes(cls, bytea, **kwargs):
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            window = cls.window(*kwargs['window'])
        else:
            window = None
        with cls.memfile(bytesio_to_bytea(bytea)) as memfile:
            with memfile.open() as f_raster:
                raster = f_raster.read(window=window, boundless=True)
                return_args = [raster]
                if kwargs.get('return_metadata', False):
                    profile = cls.metadata(f_raster, window=window)
                    return_args.append(profile)
                bands = kwargs.get('bands', None)
                if bands is not None:
                    raster = band_selection(raster, bands)
                    profile['count'] = len(bands)

                return tuple(return_args)

    @classmethod
    def save_to_disk(cls, array, file_path, **kwargs):
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
        if 'transform' not in out_meta:
            # add identity transform to suppress "NotGeoreferencedWarning"
            out_meta['transform'] = cls.driver.Affine.identity()
        bands = kwargs.get('bands', None)
        if bands is not None:
            array = band_selection(array, bands)
            out_meta['count'] = len(bands)
        with cls.driver.open(file_path, 'w', **out_meta) as dest_img:
            dest_img.write(array)

    @classmethod
    def save_to_bytes(cls, array, image_format, **kwargs):        #TODO: format
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
        if 'transform' not in out_meta:
            # add identity transform to suppress "NotGeoreferencedWarning"
            out_meta['transform'] = cls.driver.Affine.identity()
        with cls.memfile() as memfile:
            with memfile.open(**out_meta) as f_mem:
                f_mem.write(array)
        return memfile

    @classmethod
    def disk_to_bytes(cls, file_path, **kwargs):
        if 'window' in kwargs and kwargs['window'] is not None:
            # crop
            window = cls.window(*kwargs['window'])
        else:
            window = None
        with cls.driver.open(file_path, 'r') as f_raster:
            data = f_raster.read(window=window, boundless=True)
            profile, meta = f_raster.profile, f_raster.meta
        if window is not None:
            profile['width'] = window.width
            profile['height'] = window.height

        meta.update(kwargs)

        bands = kwargs.get('bands', None)
        if bands is not None:
            data = band_selection(data, bands)
            profile['count'] = len(bands)
            meta['count'] = len(bands)

        with cls.memfile() as memfile:
            with memfile.open(**meta) as rst:
                rst.write(data)
            memfile.seek(0)
            bytes_arr = memfile.read()
        if kwargs.get('return_metadata', False):
            return bytes_arr, profile
        return bytes_arr



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
    def is_loadable(cls, obj):
        #TODO: check if optimizable
        try:
            if isinstance(obj, str):
                img = cls.load_from_disk(obj)
            else:
                img = cls.load_from_bytes(obj)
            return isinstance(img, np.ndarray)
        except Exception:
            return False

    @classmethod
    def size(cls, obj):
        #TODO: check if optimizable
        if isinstance(obj, str):
            img = cls.load_from_disk(obj)
        else:
            img = cls.load_from_bytes(obj)
        return list(img.shape)

    @classmethod
    def load_from_disk(cls, file_path, **kwargs):
        data = cls.loader(file_path)
        raster = data.pixel_array
        if raster.ndim == 2:
            raster = raster[np.newaxis,...]
        return raster

    @classmethod
    def load_from_bytes(cls, bytea, **kwargs):
        bytes_io = bytea_to_bytesio(bytea)
        data = cls.loader(bytes_io)
        raster = data.pixel_array
        if raster.ndim == 2:
            raster = raster[np.newaxis,...]
        return raster

    @classmethod
    def save_to_disk(cls, array, file_path, **kwargs):
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
            print(f'Driver "{dname}" initialized.')
        except Exception as e:
            print(f'Driver "{dname}" unavailable ("{str(e)}")')
