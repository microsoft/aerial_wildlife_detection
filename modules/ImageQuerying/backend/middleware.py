'''
    2021 Benjamin Kellenberger
'''

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

    

    def grabCut(self, project, imgPath, coords, return_polygon=False, num_iter=5, strictness=2):
        '''
            Runs the GrabCut algorithm on an image in the project, identified by
            the provided imgPath, as well as a set of relative coordinates (flat
            list of x, y floats). Performs GrabCut and returns either a binary
            pixel mask of the result, or a flat list of x, y coordinates of a
            polygonized version of the mask. Parameter "strictness" controls how
            many potential/unsure areas are to be involved:
            - 0: include both potential foreground and background pixels
            - 1: include potential foreground pixels
            - 2: only include clear foreground pixels
        '''
        assert isinstance(imgPath, str), f'Invalid image path provided ("{str(imgPath)}")'
        assert isinstance(coords, list) or isinstance(coords, tuple), 'Invalid coordinates format provided'
        assert len(coords) >= 6 and not len(coords)%2, 'Invalid number of coordinate points provided'

        # get image
        imgBytes = self.fileServer.getFile(project, imgPath)
        img = drivers.load_from_bytes(imgBytes)
        img = np.transpose(img, axes=(1,2,0))
        img = img[:,:,:3]                          #TODO: get bands from render config
        img = np.ascontiguousarray(img)

        # normalize image
        sz = img.shape
        img = img.reshape((3, -1))
        mins = np.min(img, 1)[:,np.newaxis]
        maxs = np.max(img, 1)[:,np.newaxis]
        img = (img - mins)/(maxs - mins)
        img = np.fliplr(img)                # make BGR
        img = 255 * img.reshape(sz).astype(np.uint8)

        # make grayscale images BGR         TODO: extra function?
        if img.shape[2] < 3:
            img = np.repeat(img, repeats=3, axis=2)

        # initialize mask from coordinates
        coords_abs = np.array(coords, dtype=np.float32)
        coords_abs[::2] *= sz[1]
        coords_abs[1::2] *= sz[0]
        if coords_abs[-1] != coords_abs[1] or coords_abs[-2] != coords_abs[0]:
            # close polygon
            coords_abs = np.concatenate((coords_abs, coords_abs[:2]))
        mask = polygons_to_bitmask([coords_abs], height=sz[0], width=sz[1])
        mask = mask.astype(np.uint8)
        mask[mask>0] = cv2.GC_PR_FGD
        mask[mask==0] = cv2.GC_BGD
        mask = np.ascontiguousarray(mask)
        rect = [
            int(np.min(coords_abs[::2])),
            int(np.min(coords_abs[1::2])),
            int(np.max(coords_abs[::2]) - np.min(coords_abs[::2])),
            int(np.max(coords_abs[1::2]) - np.min(coords_abs[1::2]))
        ]

        bgdModel = np.zeros((1,65), dtype=np.float64)
        fgdModel = np.zeros((1,65), dtype=np.float64)

        mask, _, _ = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, num_iter, mode=cv2.GC_INIT_WITH_MASK)

        #TODO
        mask = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
        # mask_out[mask_out==cv2.GC_PR_FGD] = cv2.GC_FGD if strictness <2 else cv2.GC_BGD
        # mask_out[mask_out==cv2.GC_PR_BGD] = cv2.GC_FGD if strictness <1 else cv2.GC_BGD

        if return_polygon:
            # convert mask to polygon and keep the largest only
            polys_out = Mask(mask).polygons()
            if not len(polys_out.points):
                return None
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
            return max_poly.tolist()

        else:
            return mask.tolist()