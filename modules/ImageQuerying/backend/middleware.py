'''
    2021 Benjamin Kellenberger
'''

import json
import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from detectron2.structures.masks import polygons_to_bitmask, polygon_area
from imantics import Mask

from modules.AIWorker.backend import fileserver     #TODO: make generally accessible?
from util import drivers


class ImageQueryingMiddleware:

    def __init__(self, config, dbConnector):
        self.config = config
        self.dbConnector = dbConnector

        # initialize local file server for image retrieval
        self.fileServer = fileserver.FileServer(self.config)
    


    def _load_image_for_project(self, project, imgPath, bands='all'):
        '''
            Loads an image for a given project and path into a NumPy ndarray (as
            per image drivers). Optionally performs band selection, with
            parameter "bands":
            - 'all':        select all bands (no changement)
            - 'rgb':        select only RGB bands (also converts grayscale)
            - list/tuple:   select bands in order at given index
            Returns a NumPy ndarray of the image of size HxWxB
        '''
        assert isinstance(imgPath, str), f'Invalid image path provided ("{str(imgPath)}")'
        imgBytes = self.fileServer.getFile(project, imgPath)
        img = drivers.load_from_bytes(imgBytes)

        if bands is None or bands == 'all':
            img = np.transpose(img, axes=(1,2,0))
            return img
        
        elif isinstance(bands, list) or isinstance(bands, tuple):
            img = np.stack(
                [img[b,...] for b in bands], -1
            )
            return img
        
        elif isinstance(bands, str):
            if img.shape[0] == 1:
                # grayscale image
                img = np.stack([img[b,...] for b in [0,0,0]], -1)
                return img

            if bands.lower() == 'rgb':
                bandKeys = ('red', 'green', 'blue')
            elif bands.lower() == 'bgr':
                bandKeys = ('blue', 'green', 'red')

            # get render config for project
            numBands = img.shape[0]
            renderConfig = self.dbConnector.execute('''
                    SELECT render_config
                    FROM aide_admin.project
                    WHERE shortname = %s;
                ''', (project,), 1)
            renderConfig = json.loads(renderConfig[0]['render_config'])
            indices = renderConfig['bands']['indices']
            bands = []
            for key in bandKeys:
                bands.append(min(indices[key], numBands-1))
            img = np.stack([img[b,...] for b in bands], -1)
            return img
        
        else:
            raise Exception(f'Invalid specifier provided for bands ("{bands}").')



    @staticmethod
    def _normalize_image(img):
        #TODO: make extra routine in image drivers (also with more features)
        sz = img.shape
        img = img.reshape((-1, sz[2]))
        mins = np.min(img, 0)[np.newaxis,:]
        maxs = np.max(img, 0)[np.newaxis,:]
        img = (img - mins)/(maxs - mins)
        img = 255 * img.reshape(sz)
        img = img.astype(np.uint8)
        return img



    @staticmethod
    def _contains_point(poly, points):
        '''
            Ray tracing (point-in-polygon) algorithm; from here:
            https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
            #TODO: make extra routine somewhere?
        '''
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



    def magicWand(self, project, imgPath, seedCoords, tolerance=32, maxRadius=None, rgbOnly=False):
        '''
            Magic wand functionality: loads an image for a project and receives
            relative seed coordinates [x, y]. Finds similar pixels in the
            neighborhood of the seed coordinates that have a euclidean color
            distance of at most "tolerance" (applies to [0, 255] range of pixel
            values). The neighborhood can optionally be restricted in size by
            "maxRadius".
        '''
        assert isinstance(seedCoords, list) or isinstance(seedCoords, tuple), 'Invalid coordinates format provided'
        assert len(seedCoords) == 2, f'Invalid number of seed coordinates ({len(seedCoords)} != 2)'
        if any([s<0 or s>1 for s in seedCoords]):
            # invalid seed coordinates position
            return None
        assert isinstance(tolerance, int) or isinstance(tolerance, float), f'Invalid value for tolerance ({tolerance})'
        assert maxRadius is None or isinstance(maxRadius, int), f'Invalid value for max radius ({maxRadius})'

        # get image
        img = self._load_image_for_project(project, imgPath, bands='rgb' if rgbOnly else 'all')
        img = self._normalize_image(img)
        img = np.ascontiguousarray(img).astype(np.float32)
        sz = img.shape

        # make absolute coordinates
        seedCoords = np.round(np.array([seedCoords[0]*sz[1], seedCoords[1]*sz[0]])).astype(np.int32)

        # define mask
        if maxRadius is None:
            mask = np.ones(shape=sz[:2], dtype=np.bool8)
        else:
            mask = np.zeros(shape=sz[:2], dtype=np.bool8)
            halfRadius = np.round(maxRadius/2.0).astype(np.int32)
            box = [
                max(0, seedCoords[1]-halfRadius),
                max(0, seedCoords[0]-halfRadius),
                min(sz[0], seedCoords[1]+halfRadius),
                min(sz[1], seedCoords[0]+halfRadius)
            ]
            mask[box[0]:box[2], box[1]:box[3]] = 1

        # find and assign similar pixels
        seedVals = img[seedCoords[1], seedCoords[0], :]
        img_flat = np.reshape(img, (-1, sz[2]))
        mask_flat = mask.ravel()
        candidates = np.where(mask_flat==1)[0]
        dist = np.sqrt(np.sum(np.power(img_flat[candidates,:] - seedVals[np.newaxis,:], 2), 1))
        mask_flat[candidates] = dist <= tolerance
        mask_out = np.reshape(mask_flat, sz[:2])

        # close holes in mask
        mask_out = binary_fill_holes(mask_out)      #TODO: different structured element?

        # polygonize; keep largest polygon that contains seed coordinates
        result = None
        polys_out = Mask(mask_out).polygons()
        if len(polys_out.points):
            max_poly_area = 0
            max_poly = None
            for poly in polys_out.points:
                if len(poly) < 3:
                    continue
                if self._contains_point(poly, seedCoords[np.newaxis,:])[0]:
                    area = polygon_area(poly[:,0], poly[:,1])
                    if area > max_poly_area:
                        max_poly_area = area
                        max_poly = poly.ravel()
            
            # make relative coordinates again
            if max_poly is not None:
                max_poly = max_poly.astype(np.float32)
                max_poly[::2] /= sz[1]
                max_poly[1::2] /= sz[0]
                result = max_poly.tolist()
        return result
    


    def grabCut(self, project, imgPath, coords, return_polygon=False, num_iter=5):
        '''
            Runs the GrabCut algorithm on an image in the project, identified by
            the provided imgPath, as well as a list of lists of relative
            coordinates (each a flat list of x, y floats). Performs GrabCut and
            returns either a list of binary pixel masks of each of the given
            coordinates' result, or a list of flat lists of x, y coordinates of
            a polygonized version of the masks.
        '''
        assert isinstance(coords, list) or isinstance(coords, tuple), 'Invalid coordinates format provided'
        
        if isinstance(coords[0], int) or isinstance(coords[0], float):
            # just one polygon provided; encapsulate
            coords = [coords]
        for idx in range(len(coords)):
            numel = len(coords[idx])
            assert numel >= 6 and not numel%2, \
                f'Polygon #{idx+1}: Invalid number of coordinate points provided ({numel} < 6 or not even)'

        # get image
        img = self._load_image_for_project(project, imgPath, bands='bgr')       # OpenCV requires BGR
        img = np.ascontiguousarray(img).astype(np.float32)
        sz = img.shape

        img = self._normalize_image(img)

        # iterate through all lists of coordinates provided
        result = []

        for coord in coords:
            # initialize mask from coordinates
            coords_abs = np.array(coord, dtype=np.float32)
            coords_abs[::2] *= sz[1]
            coords_abs[1::2] *= sz[0]
            if coords_abs[-1] != coords_abs[1] or coords_abs[-2] != coords_abs[0]:
                # close polygon
                coords_abs = np.concatenate((coords_abs, coords_abs[:2]))

            mbr = [
                max(0, int(np.min(coords_abs[::2]))),
                max(0, int(np.min(coords_abs[1::2]))),
                min(sz[1], int(np.max(coords_abs[::2]))),
                min(sz[0], int(np.max(coords_abs[1::2])))
            ]
            mask = cv2.GC_BGD * np.ones(sz[:2], dtype=np.uint8)
            mask[mbr[1]:mbr[3], mbr[0]:mbr[2]] = cv2.GC_PR_BGD
            polymask = polygons_to_bitmask([coords_abs], height=sz[0], width=sz[1])
            mask[polymask>0] = cv2.GC_PR_FGD

            bgdModel = np.zeros((1,65), dtype=np.float64)
            fgdModel = np.zeros((1,65), dtype=np.float64)

            mask, _, _ = cv2.grabCut(img, mask, None, bgdModel, fgdModel, num_iter, mode=cv2.GC_INIT_WITH_MASK)
            mask_out = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)

            if return_polygon:
                # convert mask to polygon and keep the largest only
                polys_out = Mask(mask_out).polygons()
                if not len(polys_out.points):
                    result.append(None)
                else:
                    max_poly_area = 0
                    max_poly = None
                    for poly in polys_out.points:
                        if len(poly) < 3:
                            continue
                        area = polygon_area(poly[:,0], poly[:,1])
                        if area > max_poly_area:
                            max_poly_area = area
                            max_poly = poly.ravel()
                    
                    # make relative coordinates again
                    if max_poly is None:
                        result.append(None)
                    else:
                        max_poly = max_poly.astype(np.float32)
                        max_poly[::2] /= img.shape[1]
                        max_poly[1::2] /= img.shape[0]
                        result.append(max_poly.tolist())
            else:
                result.append(mask_out.tolist())
        
        return result