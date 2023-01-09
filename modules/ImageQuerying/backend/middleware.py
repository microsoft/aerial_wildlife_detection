'''
    2021-23 Benjamin Kellenberger
'''

from io import BytesIO
import json
import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.spatial.distance import cdist
from detectron2.structures.masks import polygons_to_bitmask

from modules.AIWorker.backend import fileserver     #TODO: make generally accessible?
from util import drivers
from util.drivers.imageDrivers import normalize_image
from . import regionProcessing


class ImageQueryingMiddleware:

    def __init__(self, config, dbConnector):
        self.config = config
        self.dbConnector = dbConnector

        # initialize local file server for image retrieval
        self.fileServer = fileserver.FileServer(self.config)



    def _load_image_for_project(self, project, img_path, bands='all'):
        '''
            Loads an image for a given project and path into a NumPy ndarray (as
            per image drivers). Optionally performs band selection, with
            parameter "bands":
            - 'all':        select all bands (no changement)
            - 'rgb':        select only RGB bands (also converts grayscale)
            - list/tuple:   select bands in order at given index
            Returns a NumPy ndarray of the image of size HxWxB
        '''
        assert isinstance(img_path, str), f'Invalid image path provided ("{str(img_path)}")'
        img = self.fileServer.getImage(project, img_path)

        if bands is None or bands == 'all':
            img = np.transpose(img, axes=(1,2,0))
            return img

        if isinstance(bands, (list, tuple)):
            img = np.stack(
                [img[b,...] for b in bands], -1
            )
            return img

        if isinstance(bands, str):
            if img.shape[0] == 1:
                # grayscale image
                img = np.stack([img[b,...] for b in [0,0,0]], -1)
                return img

            if bands.lower() == 'rgb':
                band_keys = ('red', 'green', 'blue')
            elif bands.lower() == 'bgr':
                band_keys = ('blue', 'green', 'red')

            # get render config for project
            num_bands = img.shape[0]
            render_config = self.dbConnector.execute('''
                    SELECT render_config
                    FROM aide_admin.project
                    WHERE shortname = %s;
                ''', (project,), 1)
            render_config = json.loads(render_config[0]['render_config'])
            indices = render_config['bands']['indices']
            bands = []
            for key in band_keys:
                bands.append(min(indices[key], num_bands-1))
            img = np.stack([img[b,...] for b in bands], -1)
            return img

        raise Exception(f'Invalid specifier provided for bands ("{bands}").')



    def magicWand(self, project, imgPath, seedCoords, tolerance=32, maxRadius=None, rgbOnly=False):
        '''
            Magic wand functionality: loads an image for a project and receives
            relative seed coordinates [x, y]. Finds similar pixels in the
            neighborhood of the seed coordinates that have a euclidean color
            distance of at most "tolerance" (applies to [0, 255] range of pixel
            values). The neighborhood can optionally be restricted in size by
            "maxRadius".
        '''
        assert isinstance(seedCoords, (list, tuple)), 'Invalid coordinates format provided'
        assert len(seedCoords) == 2, f'Invalid number of seed coordinates ({len(seedCoords)} != 2)'
        if any(s<0 or s>1 for s in seedCoords):
            # invalid seed coordinates position
            return None
        assert isinstance(tolerance, (int, float)), f'Invalid value for tolerance ({tolerance})'
        assert maxRadius is None or isinstance(maxRadius, int), f'Invalid value for max radius ({maxRadius})'

        # get image
        img = self._load_image_for_project(project, imgPath, bands='rgb' if rgbOnly else 'all')
        img = normalize_image(img, band_axis=-1)
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
        dist = np.sqrt(np.sum(np.power(img_flat[candidates,:] - seedVals[np.newaxis,:], 2), 1))     #TODO: should be np.mean, but np.sum works better somehow
        mask_flat[candidates] = dist <= tolerance
        mask_out = np.reshape(mask_flat, sz[:2])

        # close holes in mask
        mask_out = binary_fill_holes(mask_out)      #TODO: different structured element?

        # dilate to remove border effects when polygonizing
        mask_out = binary_dilation(mask_out)        #TODO: ditto?

        # polygonize; keep largest polygon that contains seed coordinates
        result = regionProcessing.mask_to_poly(mask_out, True, seedCoords)

        # make relative coordinates again
        if result is not None:
            result = result.astype(np.float32)
            result[::2] /= sz[1]
            result[1::2] /= sz[0]
            result = result.tolist()
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
        assert isinstance(coords, (list, tuple)), 'Invalid coordinates format provided'

        if isinstance(coords[0], (int, float)):
            # just one polygon provided; encapsulate
            coords = [coords]
        for idx, coord in enumerate(coords):
            numel = len(coord)
            assert numel >= 6 and not numel%2, \
                f'Polygon #{idx+1}: Invalid number of coordinate points provided ' + \
                f'({numel} < 6 or not even)'

        # get image
        img = self._load_image_for_project(project, imgPath, bands='bgr')   # OpenCV requires BGR
        img = np.ascontiguousarray(img).astype(np.float32)
        size = img.shape

        img = normalize_image(img, band_axis=1)

        # iterate through all lists of coordinates provided
        result = []

        for coord in coords:
            # initialize mask from coordinates
            coords_abs = np.array(coord, dtype=np.float32)
            coords_abs[::2] *= size[1]
            coords_abs[1::2] *= size[0]
            if coords_abs[-1] != coords_abs[1] or coords_abs[-2] != coords_abs[0]:
                # close polygon
                coords_abs = np.concatenate((coords_abs, coords_abs[:2]))

            mbr = [
                max(0, int(np.min(coords_abs[::2]))),
                max(0, int(np.min(coords_abs[1::2]))),
                min(size[1], int(np.max(coords_abs[::2]))),
                min(size[0], int(np.max(coords_abs[1::2])))
            ]
            mask = cv2.GC_BGD * np.ones(size[:2], dtype=np.uint8)
            mask[mbr[1]:mbr[3], mbr[0]:mbr[2]] = cv2.GC_PR_BGD
            polymask = polygons_to_bitmask([coords_abs], height=size[0], width=size[1])
            mask[polymask>0] = cv2.GC_PR_FGD

            bgdModel = np.zeros((1,65), dtype=np.float64)
            fgdModel = np.zeros((1,65), dtype=np.float64)

            mask, _, _ = cv2.grabCut(img, mask, None, bgdModel, fgdModel, num_iter, mode=cv2.GC_INIT_WITH_MASK)
            mask_out = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)

            # dilate to remove border effects when polygonizing
            mask_out = binary_dilation(mask_out)        #TODO: more advanced features?

            if return_polygon:
                # convert mask to polygon and keep the largest only
                largest = regionProcessing.mask_to_poly(mask_out, True)

                # make relative coordinates again
                if largest is not None:
                    largest = largest.astype(np.float32)
                    largest[::2] /= size[1]
                    largest[1::2] /= size[0]
                    largest = largest.tolist()
                result.append(largest)
            else:
                result.append(mask_out.tolist())

        return result



    def select_similar(self, project, imgPath, seed_polygon, tolerance=0.01, return_polygon=False, num_max=32):
        '''
            Receives an image and polygon within the image and tries to find
            other regions that are visually similar. The similarity is defined
            by "features" that are extracted for the seed polygon region, as
            well as "tolerance" that defines the quantile of the euclidean
            distance of the features of other regions in the image to the seed
            region. Other regions are obtained using a specified
            superpixelization algorithm, with optional arguments.
        '''
        #TODO: fixed kwargs for now
        regionMethod = 'quickshift'
        def featureCalc(img, regionMap):
            # return regionProcessing.custom_features(img, regionMap,
            #     (lambda x: np.mean(x, 0), lambda x: np.std(x, 0)))
            return np.concatenate(
                (
                    regionProcessing.custom_features(img, regionMap,
                        (lambda x: np.mean(x, 0), lambda x: np.std(x, 0))),
                    regionProcessing.histogram_of_colors(img, regionMap, num_bins=25)
                ),
                1
            )
            # return regionProcessing.histogram_of_colors(img, regionMap, num_bins=25)
            # return np.concatenate(
            #     (
            #         regionProcessing.histogram_of_colors(img, regionMap, num_bins=20),
            #         regionProcessing.bag_of_visual_words(img, regionMap, k=10, num_bins=20)
            #     ),
            #     1
            # )

        assert isinstance(seed_polygon, (list, tuple)), 'Invalid coordinates format provided'
        numel = len(seed_polygon)
        assert numel >= 6 and not numel%2, \
            f'Invalid number of coordinate points provided ({numel} < 6 or not even)'

        # get image
        img = self._load_image_for_project(project, imgPath, bands='rgb')
        img = np.ascontiguousarray(img).astype(np.float32)
        img = normalize_image(img, band_axis=-1)
        sz = img.shape

        # get mask for seed region
        coords_abs = np.array(seed_polygon, dtype=np.float32)
        coords_abs[::2] *= sz[1]
        coords_abs[1::2] *= sz[0]
        if coords_abs[-1] != coords_abs[1] or coords_abs[-2] != coords_abs[0]:
            # close polygon
            coords_abs = np.concatenate((coords_abs, coords_abs[:2]))
        polymask = polygons_to_bitmask([coords_abs], height=sz[0], width=sz[1])

        # segment image into regions
        regions = regionProcessing.find_regions(img, regionMethod)      #TODO: kwargs
        rIdx = np.sort(np.unique(regions))

        # calculate features for seed and candidate regions and compare
        candidateFeatures = featureCalc(img, regions)   #TODO: remove candidates intersecting with seed region
        seedFeatures = []
        for r in rIdx:
            mask = regions == r
            if np.sum(mask*polymask):
                seedFeatures.append(candidateFeatures[r,:])
        seedFeatures = np.array(seedFeatures)

        # candidateFeatures = featureCalc(img, regions)                               #TODO: remove candidates intersecting with seed region
        dists = cdist(seedFeatures, candidateFeatures)                              #TODO: l2 distance suboptimal for histograms; use Wasserstein metric instead?
        # dists = np.zeros(len(candidateFeatures))
        # for c in range(len(candidateFeatures)):                                     #TODO: speedup?
        #     dists[c] = wasserstein_distance(candidateFeatures[c,:], seedFeatures)
        dists = np.mean(dists, 0)
        tolQuant = np.quantile(dists, tolerance)                                    #TODO: feature distances are unbounded; need to adjust tolerance somehow
        valid = np.where(dists <= tolQuant)[0]

        mask_result = np.zeros_like(regions)
        for v in valid:                                     #TODO: speedup?
            mask_result[regions==rIdx[v]] = 1

        # extract polygons and keep those with areas most similar to seed area
        polygons = regionProcessing.mask_to_poly(mask_result, False)
        areas = np.array([regionProcessing.polygon_area(p[:,0], p[:,1]) for p in polygons])
        seedArea = regionProcessing.polygon_area(coords_abs[::2], coords_abs[1::2])
        order = np.argsort(np.power(areas - seedArea, 2))
        order = order[:min(len(order), num_max)]

        result = []
        for o in order:
            # make relative coordinates again
            poly = polygons[o].astype(np.float32)
            poly[:,0] /= sz[1]
            poly[:,1] /= sz[0]
            result.append(poly.ravel().tolist())

        return result



    def get_image_metadata(self, upload):
        '''
            Receives a Bottle.py upload of a single file and tries to parse it as an image. If it is
            parseable, returns metadata (including CRS if available and band names [a.k.a.
            "descriptions"]); else returns None.
        '''
        try:
            meta = {}
            file = next(iter(upload.values()))
            bio = BytesIO(file.file.read())
            driver = drivers.get_driver(bio)
            if driver is None:
                # unreadable file
                return None
            try:
                meta_raw = driver.metadata(bio)
                if 'descriptions' in meta_raw:
                    meta['descriptions'] = meta_raw['descriptions']
                else:
                    # no descriptions (band names) provided; fallback
                    size = driver.size(bio)
                    meta['descriptions'] =  [f'Band {s+1}' for s in range(size[0])]
                if meta_raw.get('crs', None) is not None:
                    #TODO: check if CRS is compatible with PostGIS
                    meta['crs'] = str(meta_raw['crs'])
            except Exception:
                # driver does not support metadata querying; fallback
                size = driver.size(bio)
                bands = [f'Band {s+1}' for s in range(size[0])]
                meta = {
                    'descriptions': bands,
                    'crs': None
                }
            return meta

        except Exception:
            return None
