'''
    Contains functionality to split a NumPy ndarray image into shards (patches)
    on a regular grid.

    2020-23 Benjamin Kellenberger
'''

import os
import numpy as np
from tqdm import tqdm
from celery import current_task

from util import drivers
from util.drivers.imageDrivers import normalize_image
drivers.init_drivers()



def get_split_positions(array, patch_size, stride=None, tight=True,
                        discard_homogeneous_percentage=None,
                        discard_homogeneous_quantization_value=255, celery_update_interval=-1):
    '''
        Receives one of:
            - a NumPy ndarray of size (BxHxW)
            - a str, denoting the file path of an image
            - a bytes or BytesIO object, containing image data
        and returns a list of [top, left, height, width] coordinates of split patches. The splitting
        raster can be customized through the parameters. Inputs:
            - array:        The NumPy ndarray to be split into patches.
            - patch_size:   Either an int or a tuple of (height, width) of
                            the patch dimensions.
            - stride:       The offsets of each patch with respect to its
                            immediate neighbor. Can be one of the following: - None: strides are set
                            to the values in "patch_size" - int:  equal stride in both height and
                            width direction - tuple of (height, width) ints for both direction
            - tight:        If True, the last patches in x and y direction might
                            be shifted towards the left (resp. top) if needed, so that none of the
                            patches exceeds the image boundaries. If False, patches might exceed the
                            image boundaries and contain black borders (filled with all-zeros).
            - discard_homogeneous_percentage:
                            If float or int, any patch with this or more percent of pixels that have
                            the same values across all bands will be discarded. Useful to get rid of
                            e.g. bordering patches in slanted satellite stripes. If <= 0, >= 100 or
                            None, no patch will be discarded.
            - discard_homogeneous_quantization_value:
                            Int in [1, 255]. Defines the number of color bins used for quantization
                            during pixel homogeneity evaluation. Smaller values clump more diverse
                            pixels together; larger values require pixels to be more similar to be
                            considered identical.

        Returns:
            - coords:       A list of N tuples containing the (top, left, height, width) pixel
                            coordinates of the top left corner of the patches.
    '''
    if not isinstance(array, np.ndarray):
        # get image size from disk or bytes
        driver = drivers.get_driver(array)
        size = driver.size(array)
    else:
        size = array.shape

    # assertions
    assert len(size) == 3, \
        f'Invalid number of dimensions for input, got {len(size)}, expected 3 (BxWxH)'
    if isinstance(patch_size, int):
        patch_size = min(patch_size, max(size[1], size[2]))
        patch_size = (patch_size, patch_size)
    elif isinstance(patch_size, (tuple, list)):
        if len(patch_size)==1:
            patch_size = (patch_size, patch_size)
        assert isinstance(patch_size[0], int), f'"{str(patch_size[0])}" is not an integer value.'
        assert isinstance(patch_size[1], int), f'"{str(patch_size[1])}" is not an integer value.'
        # need to flip patch size as NumPy indexes height first
        patch_size = (max(1,min(patch_size[1], size[1])), max(1,min(patch_size[0], size[2])))
    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = min(stride, max(size[1], size[2]))
        stride = (stride, stride)
    elif isinstance(stride, (tuple, list)):
        if len(stride)==1:
            stride = (stride, stride)
        assert isinstance(stride[0], int), f'"{str(stride[0])}" is not an integer value.'
        assert isinstance(stride[1], int), f'"{str(stride[1])}" is not an integer value.'
        # ditto for stride
        stride = (max(1,min(stride[1], size[1])), max(1,min(stride[0], size[2])))
    if isinstance(discard_homogeneous_percentage, (float, int)):
        discard_homogeneous_percentage = max(0.001, min(100, discard_homogeneous_percentage))
    else:
        discard_homogeneous_percentage = None
    if not isinstance(discard_homogeneous_quantization_value, (float, int)):
        discard_homogeneous_quantization_value = 255
    else:
        discard_homogeneous_quantization_value = max(1, \
            min(255, discard_homogeneous_quantization_value))

    if not hasattr(current_task, 'update_state'):
        celery_update_interval = -1
    elif celery_update_interval >= 0:
        celery_update_msg = 'determining view locations'
        if isinstance(array, str):
            celery_update_msg += f' ("{array}")'

    # define crop locations
    x_locs = list(range(0, size[2], stride[1]))
    y_locs = list(range(0, size[1], stride[0]))
    if tight:
        # limit to image borders
        max_x = size[2] - patch_size[1]
        max_y = size[1] - patch_size[0]
        x_locs[-1] = max_x
        y_locs[-1] = max_y
    else:
        # add extra steps if required
        while x_locs[-1] + patch_size[1] < size[2]:
            x_locs.append(x_locs[-1] + stride[1])
        while y_locs[-1] + patch_size[0] < size[1]:
            y_locs.append(y_locs[-1] + stride[0])

    if len(x_locs) <= 1 and len(y_locs) <= 1:
        # patch size is greater than image size; return image
        return [(0, 0, size[1], size[2])]

    # do the cropping
    num_coords = len(x_locs)*len(y_locs)
    tbar = tqdm(range(num_coords))
    if discard_homogeneous_percentage is not None:
        print('Locating split positions, discarding empty patches...')
    coords = []
    count = 0
    for x_loc in x_locs:
        for y_loc in y_locs:
            if celery_update_interval >= 0 and count % celery_update_interval != 0:
                current_task.update_state(
                    meta={
                        'message': celery_update_msg,
                        'done': count,
                        'total': num_coords
                    }
                )
            count += 1
            pos = (int(y_loc), int(x_loc))
            end = (min(pos[0]+patch_size[0], size[1]), min(pos[1]+patch_size[1], size[2]))
            if discard_homogeneous_percentage is not None:
                # get most frequent pixel value across bands and check percentage of it
                if isinstance(array, np.ndarray):
                    arr_patch = array[:,pos[0]:end[0],pos[1]:end[1]]
                else:
                    arr_patch = driver.load(array, \
                                        window=[pos[1], pos[0], end[1]-pos[1], end[0]-pos[0]])
                if discard_homogeneous_percentage > 99.999999:
                    # looking for a single value in images only; speed up process
                    if all(len(np.unique(arr_patch[a,...]))==1 for a in range(len(arr_patch))):
                        tbar.update(1)
                        continue

                else:
                    psz = arr_patch.shape
                    arr_patch_int = normalize_image(arr_patch,
                                        color_range=discard_homogeneous_quantization_value)
                    _, counts = np.unique(np.reshape(arr_patch_int, (psz[0], -1)),
                                        axis=1, return_counts=True)
                    if np.max(counts) > discard_homogeneous_percentage/100.0*(psz[1]*psz[2]):
                        tbar.update(1)
                        continue
            coord = (pos[0], pos[1], end[0]-pos[0], end[1]-pos[1])
            coords.append(coord)
            tbar.update(1)

    if discard_homogeneous_percentage is not None:
        tbar.close()

    return coords



def split_image(array, patch_size, stride=None, tight=True, discard_homogeneous_percentage=None,
                discard_homogeneous_quantization_value=255, save_root=None, return_patches=True,
                celery_update_interval=-1):
    '''
        Receives one of:
            - a NumPy ndarray of size (BxHxW)
            - a str, denoting the file path of an image
            - a bytes or BytesIO object, containing image data
        and splits it into patches on a regular grid. The splitting raster can
        be customized through the parameters. Inputs:
            - array:        The NumPy ndarray to be split into patches.
            - patch_size:   Either an int or a tuple of (width, height) of
                            the patch dimensions.
            - stride:       The offsets of each patch with respect to its
                            immediate neighbor. Can be one of the following: -
                            None: strides are set to the values in "patch_size" -
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

    if not hasattr(current_task, 'update_state'):
        celery_update_interval = -1
    elif celery_update_interval >= 0:
        celery_update_msg = 'splitting image'
        if isinstance(array, str):
            celery_update_msg += f' ("{array}")'

    coords = get_split_positions(array, patch_size, stride, tight,
                                discard_homogeneous_percentage,
                                discard_homogeneous_quantization_value,
                                celery_update_interval=celery_update_interval)

    if not isinstance(array, np.ndarray):
        # get image size from disk or bytes
        driver = drivers.get_driver(array)
        size = driver.size(array)
    else:
        size = array.shape

    if isinstance(save_root, str):
        root_filename, ext = os.path.splitext(save_root)
        parent, _ = os.path.split(save_root)
        os.makedirs(parent, exist_ok=True)

    # do the cropping
    patches = []
    filenames = []
    for idx, pos in enumerate(tqdm(coords)):
        end = (min(pos[0]+patch_size[1], size[1]), min(pos[1]+patch_size[0], size[2]))
        arr_patch = driver.load(array, window=[pos[1], pos[0], end[1]-pos[1], end[0]-pos[0]])
        if return_patches:
            #TODO: save to disk already here to prevent memory overflow with large images
            patches.append(arr_patch)
        if save_root is not None:
            # get new patch filename
            fname = f'{root_filename}_{pos[1]}_{pos[0]}{ext}'
            driver.save_to_disk(arr_patch, fname)
            filenames.append(fname)

        if celery_update_interval >= 0 and idx+1 % celery_update_interval != 0:
            current_task.update_state(
                meta={
                    'message': celery_update_msg,
                    'done': idx+1,
                    'total': len(coords)
                }
            )

    return_args = [coords]
    if return_patches:
        return_args.append(patches)
    if save_root is not None:
        return_args.append(filenames)
    return tuple(return_args)
