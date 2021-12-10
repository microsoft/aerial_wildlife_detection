'''
    2021 Benjamin Kellenberger
'''

import json
import numpy as np
import cv2
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

    

    def grabCut(self, project, imgPath, coords, return_polygon=False, num_iter=5):
        '''
            Runs the GrabCut algorithm on an image in the project, identified by
            the provided imgPath, as well as a list of lists of relative
            coordinates (each a flat list of x, y floats). Performs GrabCut and
            returns either a list of binary pixel masks of each of the given
            coordinates' result, or a list of flat lists of x, y coordinates of
            a polygonized version of the masks.
        '''
        assert isinstance(imgPath, str), f'Invalid image path provided ("{str(imgPath)}")'
        assert isinstance(coords, list) or isinstance(coords, tuple), 'Invalid coordinates format provided'
        
        if isinstance(coords[0], int) or isinstance(coords[0], float):
            # just one polygon provided; encapsulate
            coords = [coords]
        for idx in range(len(coords)):
            numel = len(coords[idx])
            assert numel >= 6 and not numel%2, \
                f'Polygon #{idx+1}: Invalid number of coordinate points provided ({numel} < 6 or not even)'

        # get image
        imgBytes = self.fileServer.getFile(project, imgPath)
        img = drivers.load_from_bytes(imgBytes)

        # make grayscale images BGR         TODO: extra function?
        if img.shape[0] < 3:
            img = np.repeat(img, repeats=3, axis=0)

        # get render config for project
        numBands = img.shape[0]
        try:
            renderConfig = self.dbConnector.execute('''
                    SELECT render_config
                    FROM aide_admin.project
                    WHERE shortname = %s;
                ''', (project,), 1)
            renderConfig = json.loads(renderConfig[0]['render_config'])
            indices = renderConfig['bands']['indices']
            bands = []
            for key in ('blue', 'green', 'red'):        # OpenCV requires BGR mode
                bands.append(min(indices[key], numBands-1))
        except:
            bands = (0, min(1,numBands-1), min(2,numBands-1))

        img = np.stack([img[b,:,:] for b in bands], -1)
        img = np.ascontiguousarray(img).astype(np.float32)
        sz = img.shape

        # normalize image
        img = img.reshape((-1, 3))
        mins = np.min(img, 0)[np.newaxis,:]
        maxs = np.max(img, 0)[np.newaxis,:]
        img = (img - mins)/(maxs - mins)
        img = 255 * img.reshape(sz)
        img = img.astype(np.uint8)

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
                        if len(poly) < 6:
                            continue
                        area = polygon_area(poly[:,0], poly[:,1])
                        if area > max_poly_area:
                            max_poly_area = area
                            max_poly = poly.ravel()
                    
                    # make relative coordinates again
                    max_poly = max_poly.astype(np.float32)
                    max_poly[::2] /= img.shape[1]
                    max_poly[1::2] /= img.shape[0]
                    result.append(max_poly.tolist())

            else:
                result.append(mask_out.tolist())
        
        return result