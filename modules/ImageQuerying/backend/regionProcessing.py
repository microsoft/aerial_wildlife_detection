'''
    Wrappers around image region utilities.

    2021 Benjamin Kellenberger
'''

from collections.abc import Iterable
import numpy as np
from skimage import segmentation
from sklearn.cluster import KMeans
from imantics import Mask
from detectron2.structures.masks import polygon_area


def find_regions(image, methodName, methodKwargs={}):
    '''
        Applies a segmentation algorithm as provided by the skimage library on
        an image.
    '''
    assert hasattr(segmentation, methodName), f'Invalid region creation method "{methodName}".'
    method = getattr(segmentation, methodName)
    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    return method(image, **methodKwargs)



def histogram_of_colors(image, regionMap, num_bins=10):
    '''
        Calculates a histogram of colors on an image for each region in a given
        region map. The image can have an arbitrary number of bands (HxWxB) and
        may consist of pixel-wise features in the first place.
    '''
    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    sz = image.shape

    img_flat = np.reshape(image, (-1, sz[2]))
    rmap_flat = regionMap.ravel().astype(np.int32)
    regionIdx = np.sort(np.unique(rmap_flat))
    hoc = np.zeros(shape=(len(regionIdx), num_bins))

    #TODO: speedup using numba?
    for r in regionIdx:
        mask = rmap_flat==r
        hoc[r,:] = np.histogram(img_flat[mask,:], num_bins, range=(0,255))[0].astype(np.float32) / mask.sum()
    return hoc



def bag_of_visual_words(image, regionMap, k, num_bins=10):
    '''
        Calculates a BovW histogram on an image for each region in a given
        region map. The image can have an arbitrary number of bands (HxWxB) and
        may consist of pixel-wise features in the first place.
    '''
    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    sz = image.shape

    img_flat = np.reshape(image, (-1, sz[2]))
    rmap_flat = regionMap.ravel().astype(np.int32)
    regionIdx = np.sort(np.unique(rmap_flat))

    bovwH = np.zeros(shape=(len(regionIdx), num_bins))
    img_clu = KMeans(k).fit(img_flat).labels_

    #TODO: speedup using numba?
    for r in regionIdx:
        mask = rmap_flat==r
        bovwH[r,:] = np.histogram(img_clu[mask], num_bins, range=(0,num_bins))[0].astype(np.float32) / mask.sum()
    return bovwH



def custom_features(image, regionMap, featureFuns):
    '''
        Receives an image, region map, as well as one or more functions that
        return a vector of features for each region, and applies them to the
        image.
    '''
    if not isinstance(featureFuns, Iterable):
        featureFuns = (featureFuns,)
    featureFuns = list(featureFuns)

    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    sz = image.shape

    img_flat = np.reshape(image, (-1, sz[2]))
    rmap_flat = regionMap.ravel().astype(np.int32)
    regionIdx = np.sort(np.unique(rmap_flat))

    out = []

    #TODO: speedup using numba?
    for r in regionIdx:
        mask = rmap_flat==r
        out.append(np.concatenate([f(img_flat[mask]) for f in featureFuns]))
    return np.array(out)




def contains_point(poly, points):
    '''
        Ray tracing (point-in-polygon) algorithm; from here:
        https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    '''
    if points.ndim == 1:
        points = points[np.newaxis,:]
    x, y = points[:,0], points[:,1]
    n = len(poly)
    inside = np.zeros(len(x),np.bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y,p2y)) & (y <= max(p1y,p2y)) & (x <= max(p1x,p2x)))[0]
        if p1y != p2y:
            xints = (y[idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]    

        p1x,p1y = p2x,p2y
    return inside



def mask_to_poly(mask, largest_only=False, mustContain=None):
    '''
        Transforms a binary mask into polygons. If "largest_only" is True,
        returns only the polygon with the largest area. Parameter "mustContain"
        may be an ndarray of x,y coordinates that the polygon must contain to be
        considered, or None if this does not apply.
    '''
    polys_out = Mask(mask).polygons()
    if not len(polys_out.points):
        return None
    else:
        result = []
        max_poly_area = 0
        max_poly = None
        for poly in polys_out.points:
            if len(poly) < 3:
                continue
            if mustContain is not None and not all(contains_point(poly, mustContain)):
                continue
            if largest_only:
                area = polygon_area(poly[:,0], poly[:,1])
                if area > max_poly_area:
                    max_poly_area = area
                    max_poly = poly.ravel()
            else:
                result.append(poly)
    
        if largest_only:
            return max_poly
        else:
            return result