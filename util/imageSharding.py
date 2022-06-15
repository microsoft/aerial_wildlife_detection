'''
    Contains functionality to split a NumPy ndarray image into shards (patches)
    on a regular grid.

    2020-22 Benjamin Kellenberger
'''

import os
import numpy as np

from util import drivers
drivers.init_drivers()
from util.drivers.imageDrivers import normalize_image



def get_split_positions(array, patchSize, stride=None, tight=True, discard_homogeneous_percentage=None, discard_homogeneous_quantization_value=255):
    '''
        Receives one of:
            - a NumPy ndarray of size (BxWxH)
            - a str, denoting the file path of an image
            - a bytes or BytesIO object, containing image data
        and returns a list of [left, top, width, height] coordinates of split
        patches. The splitting raster can be customized through the parameters.
        Inputs:
            - array:        The NumPy ndarray to be split into patches.
            - patchSize:    Either an int or a tuple of (width, height) of
                            the patch dimensions.
            - stride:       The offsets of each patch with respect to its
                            immediate neighbor. Can be one of the following: -
                            None: strides are set to the values in "patchSize" -
                            int:  equal stride in both x and y direction - tuple
                            of (x, y) ints for both direction
            - tight:        If True, the last patches in x and y direction might
                            be shifted towards the left (resp. top) if needed,
                            so that none of the patches exceeds the image
                            boundaries. If False, patches might exceed the image
                            boundaries and contain black borders (filled with
                            all-zeros).
            - discard_homogeneous_percentage:
                            If float or int, any patch with this or more percent
                            of pixels that have the same values across all bands
                            will be discarded. Useful to get rid of e.g.
                            bordering patches in slanted satellite stripes. If
                            <= 0, >= 100 or None, no patch will be discarded.
            - discard_homogeneous_quantization_value:
                            Int in [1, 255]. Defines the number of color bins
                            used for quantization during pixel homogeneity
                            evaluation. Smaller values clump more diverse pixels
                            together; larger values require pixels to be more
                            similar to be considered identical.
        
        Returns:
            - coords:       A list of N tuples containing the (left, top, width,
                            height) pixel coordinates of the top left corner of
                            the patches.
    '''
    if not isinstance(array, np.ndarray):
        # get image size from disk or bytes
        driver = drivers.get_driver(array)
        sz = driver.size(array)
    else:
        sz = array.shape
    
    # assertions
    assert len(sz) == 3, f'Invalid number of dimensions for input, got {len(sz)}, expected 3 (BxWxH)'
    if isinstance(patchSize, int):
        patchSize = min(patchSize, max(sz[1], sz[2]))
        patchSize = (patchSize, patchSize)
    elif isinstance(patchSize, tuple) or isinstance(patchSize, list):
        if len(patchSize)==1:
            patchSize = (patchSize, patchSize)
        assert isinstance(patchSize[0], int), f'"{str(patchSize[0])}" is not an integer value.'
        assert isinstance(patchSize[1], int), f'"{str(patchSize[1])}" is not an integer value.'
        # need to flip patch size as NumPy indexes height first
        patchSize = (max(1,min(patchSize[1], sz[1])), max(1,min(patchSize[0], sz[2])))
    if stride is None:
        stride = patchSize
    elif isinstance(stride, int):
        stride = min(stride, max(sz[1], sz[2]))
        stride = (stride, stride)
    elif isinstance(stride, tuple) or isinstance(stride, list):
        if len(stride)==1:
            stride = (stride, stride)
        assert isinstance(stride[0], int), f'"{str(stride[0])}" is not an integer value.'
        assert isinstance(stride[1], int), f'"{str(stride[1])}" is not an integer value.'
        # ditto for stride
        stride = (max(1,min(stride[1], sz[1])), max(1,min(stride[0], sz[2])))
    if isinstance(discard_homogeneous_percentage, float) or isinstance(discard_homogeneous_percentage, int):
        if discard_homogeneous_percentage <= 0 or discard_homogeneous_percentage >= 100:
            discard_homogeneous_percentage = None
    else:
        discard_homogeneous_percentage = None
    if not isinstance(discard_homogeneous_quantization_value, int) and not isinstance(discard_homogeneous_quantization_value, float):
        discard_homogeneous_quantization_value = 255
    else:
        discard_homogeneous_quantization_value = max(1, min(255, discard_homogeneous_quantization_value))

    # define crop locations
    xLoc = list(range(0, sz[1], stride[0]))
    yLoc = list(range(0, sz[2], stride[1]))
    if tight:
        # limit to image borders
        maxX = sz[1] - patchSize[0]
        maxY = sz[2] - patchSize[1]
        xLoc[-1] = maxX
        yLoc[-1] = maxY
    else:
        # add extra steps if required
        while xLoc[-1] + patchSize[0] < sz[1]:
            xLoc.append(xLoc[-1] + stride[0])
        while yLoc[-1] + patchSize[1] < sz[2]:
            yLoc.append(yLoc[-1] + stride[1])
    
    if len(xLoc) <= 1 and len(yLoc) <= 1:
        # patch size is greater than image size; return image
        return [(0, 0, sz[1], sz[2])]
    
    # do the cropping
    coords = []
    for x in range(len(xLoc)):
        for y in range(len(yLoc)):
            pos = (int(xLoc[x]), int(yLoc[y]))
            end = (min(pos[0]+patchSize[0], sz[1]), min(pos[1]+patchSize[1], sz[2]))
            if discard_homogeneous_percentage is not None:
                # get most frequent pixel value across bands and check percentage of it
                if isinstance(array, np.ndarray):
                    arr_patch = array[:,pos[0]:end[0],pos[1]:end[1]]
                else:
                    arr_patch = driver.load(array, window=[pos[1], pos[0], end[1]-pos[1], end[0]-pos[0]])
                psz = arr_patch.shape
                arr_patch_int = normalize_image(arr_patch, color_range=discard_homogeneous_quantization_value)
                _, counts = np.unique(np.reshape(arr_patch_int, (psz[0], -1)), axis=1, return_counts=True)
                if np.max(counts) > discard_homogeneous_percentage/100.0*(psz[1]*psz[2]):
                    continue
            coord = (pos[0], pos[1], end[0]-pos[0], end[1]-pos[1])
            coords.append(coord)
    
    return coords



def split_image(array, patchSize, stride=None, tight=True, discard_homogeneous_percentage=None, discard_homogeneous_quantization_value=255, save_root=None, return_patches=True):
    '''
        Receives one of:
            - a NumPy ndarray of size (BxWxH)
            - a str, denoting the file path of an image
            - a bytes or BytesIO object, containing image data
        and splits it into patches on a regular grid. The splitting raster can
        be customized through the parameters. Inputs:
            - array:        The NumPy ndarray to be split into patches.
            - patchSize:    Either an int or a tuple of (width, height) of
                            the patch dimensions.
            - stride:       The offsets of each patch with respect to its
                            immediate neighbor. Can be one of the following: -
                            None: strides are set to the values in "patchSize" -
                            int:  equal stride in both x and y direction - tuple
                            of (x, y) ints for both direction
            - tight:        If True, the last patches in x and y direction might
                            be shifted towards the left (resp. top) if needed,
                            so that none of the patches exceeds the image
                            boundaries. If False, patches might exceed the image
                            boundaries and contain black borders (filled with
                            all-zeros).
            - discard_homogeneous_percentage:
                            If float or int, any patch with this or more percent
                            of pixels that have the same values across all bands
                            will be discarded. Useful to get rid of e.g.
                            bordering patches in slanted satellite stripes. If
                            <= 0, >= 100 or None, no patch will be discarded.
            - discard_homogeneous_quantization_value:
                            Int in [1, 255]. Defines the number of color bins
                            used for quantization during pixel homogeneity
                            evaluation. Smaller values clump more diverse pixels
                            together; larger values require pixels to be more
                            similar to be considered identical.
            - save_root:
                            Str, file name to save patches to.
                            If None (default), patches will not be saved to
                            disk. Naming format is "<img name>_<x>_<y><ext>",
                            where "<img name>" can also be a subdirectory. Also
                            returns a list of file names in this case.
            - return_patches:
                            Bool, set to True to load patches into a list of
                            NumPy ndarrays. Cannot be False if "save_root" is
                            None.
        
        Returns:
            - patches:      A list of N NumPy ndarrays containing all the
                            patches cropped from the input "array". Only
                            returned if "return_patches" is True.
            - coords:       A list of N tuples containing the (x, y) pixel
                            coordinates of the top left corner of the patches.
    '''
    assert (isinstance(save_root, str) and isinstance(array, str)) or return_patches, \
        'either "return_patches" must be True or "save_root" defined'

    coords = get_split_positions(array, patchSize, stride, tight,
                                discard_homogeneous_percentage, discard_homogeneous_quantization_value)
    
    if not isinstance(array, np.ndarray):
        # get image size from disk or bytes
        driver = drivers.get_driver(array)
        sz = driver.size(array)
    else:
        sz = array.shape

    if isinstance(save_root, str):
        rootFilename, ext = os.path.splitext(save_root)
        parent, _ = os.path.split(save_root)
        os.makedirs(parent, exist_ok=True)

    # do the cropping
    patches = []
    filenames = []
    for pos in coords:
        end = (min(pos[0]+patchSize[0], sz[1]), min(pos[1]+patchSize[1], sz[2]))
        arr_patch = driver.load(array, window=[pos[1], pos[0], end[1]-pos[1], end[0]-pos[0]])
        # arr_patch = array[:,pos[0]:end[0],pos[1]:end[1]]
        if discard_homogeneous_percentage is not None:
            # get most frequent pixel value across bands and check percentage of it
            psz = arr_patch.shape
            arr_patch_int = normalize_image(arr_patch, color_range=discard_homogeneous_quantization_value)
            _, counts = np.unique(np.reshape(arr_patch_int, (psz[0], -1)), axis=1, return_counts=True)
            if np.max(counts) > discard_homogeneous_percentage/100.0*(psz[1]*psz[2]):
                continue
        # patch = np.zeros(shape=(sz[0], *patchSize), dtype=array.dtype)
        # patch[:,:(end[0]-pos[0]),:(end[1]-pos[1])] = arr_patch
        # patches.append(patch)
        if return_patches:
            patches.append(arr_patch)       #TODO: save to disk already here to prevent memory overflow with large images
        if save_root is not None:
            # get new patch filename
            fname = f'{rootFilename}_{pos[0]}_{pos[1]}{ext}'
            driver.save_to_disk(arr_patch, fname)
            filenames.append(fname)

    returnArgs = [coords]
    if return_patches:
        returnArgs.append(patches)
    if save_root is not None:
        returnArgs.append(filenames)
    return tuple(returnArgs)