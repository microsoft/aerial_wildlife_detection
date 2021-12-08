'''
    Contains functionality to split a NumPy ndarray image into shards (patches)
    on a regular grid.

    2020-21 Benjamin Kellenberger
'''

import numpy as np


def split_image(array, patchSize, stride=None, tight=True):
    '''
        Receives a NumPy ndarray of size (BxWxH) and splits it into patches
        on a regular grid.
        The splitting raster can be customized through the parameters.
        Inputs:
            - array:        The NumPy ndarray to be split into patches.
            - patchSize:    Either an int or a tuple of (width, height) of
                            the patch dimensions.
            - stride:       The offsets of each patch with respect to its
                            immediate neighbor. Can be one of the following:
                            - None: strides are set to the values in "patchSize"
                            - int:  equal stride in both x and y direction
                            - tuple of (x, y) ints for both direction
            - tight:        If True, the last patches in x and y direction might
                            be shifted towards the left (resp. top) if needed, so
                            that none of the patches exceeds the image boundaries.
                            If False, patches might exceed the image boundaries and
                            contain black borders (filled with all-zeros).
        
        Returns:
            - patches:      A list of N NumPy ndarrays containing all the patches cropped
                            from the input "array".
            - coords:       A list of N tuples containing the (x, y) pixel coordinates
                            of the top left corner of the patches.
    '''

    # assertions
    assert isinstance(array, np.ndarray), 'Input is not a NumPy ndarray.'
    sz = array.shape
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
        return [array], [(0,0)]
    
    # do the cropping
    patches = []
    coords = []
    for x in range(len(xLoc)):
        for y in range(len(yLoc)):
            pos = (int(xLoc[x]), int(yLoc[y]))
            end = (min(pos[0]+patchSize[0], sz[1]), min(pos[1]+patchSize[1], sz[2]))
            patch = np.zeros(shape=(sz[0], *patchSize), dtype=array.dtype)
            patch[:,:(end[0]-pos[0]),:(end[1]-pos[1])] = array[:,pos[0]:end[0],pos[1]:end[1]]
            # patch = image.crop((pos[0], pos[1], pos[0]+patchSize[0], pos[1]+patchSize[1]))
            patches.append(patch)
            coords.append(pos)
    
    return patches, coords