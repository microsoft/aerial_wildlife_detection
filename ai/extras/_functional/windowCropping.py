'''
    2019 Benjamin Kellenberger
'''
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


class WindowCropper:
    '''
        Takes a full-sized image and bounding box labels and splits it up into patches.
        Parameter 'cropMode' defines the way the image is split:
        - 'strided': patches are created on a regular grid.
        - 'objectCentered': patches are cropped around each object.
        - 'windowCropping': patches will be cropped so that they include as many objects as possible
                            each (all objects will appear in at least one patch).

        For 'strided' option:
            Parameter 'stride' can be set to values smaller than patchSize to allow for extra-large
            sets of patches. Defaults to patchSize.
            Parameters 'minBBoxArea' and 'minBBoxAreaFrac' control the bounding boxes carried over to
            the patches. Due to the cropping on a regular grid it might happen that a bounding box gets
            cut in size. These parameters avoid spurious boxes at the patch borders that hardly contain
            an object anymore. 'minBBoxArea' denotes the minimum area (in pixels) a box must have to be
            carried over; 'minBBoxAreaFrac' does the same, but as a fraction of the original box area.
        
        For 'objectCentered' option:
            Parameter 'cropSize' denotes the patch size cropped around the object. Note that the patch
            will eventually always be resized to 'patchSize'. May either be a tuple/list of two values for
            x and y, or else a single value for square patches.
            If floating point value(s) are provided, the crop will be made with respect to the object's size.
            For example, a value of (1.5, 1.2) will crop a patch that is 1.5 times the object's width and 1.2
            times its height, then rescaled to 'patchSize'. Note that the minimum and maximum sizes (e.g. for
            particularly large or small objects) can also be set with the two parameters below.
            Defaults to None (i.e., identical to the 'patchSize' parameter).
            'minCropSize': minimum size in pixels for a patch (list/tuple of ints or single int). Defaults to
            None (i.e., do not constrain).
            'maxCropSize': maximum size in pixels for a patch (list/tuple of ints or single int). Defaults to
            None (i.e., do not constrain).
            If 'forcePatchSizeAspectRatio' is set to True, uneven patches around bounding boxes (i.e., scaled
            by a multiplier of the bounding box's dimensions) will be forced to be of 'patchSize's aspect ratio.
            If 'maintainAspectRatio' is set to True, patches around bounding boxes will retain the bounding box's
            width / height ratio, unless constrained by maxCropSize. Otherwise the patches will be transformed
            to match the 'patchSize' specifier. Has no effect if 'forcePatchSize' is set to True.

        For 'windowCropping' option:
            Patches of size 'patchSize' will be evaluated in a search radius around each object and chosen so
            that they include the object and as many other ones of the still uncovered objects as possible.
            The included ones will then be removed from the pool, and the process is repeated for the remaining
            objects until all are covered.
            Parameter 'searchStride' controls the spacing of the search windows around the respective bounding box.
            Smaller values might give finer precision, but take quadratically longer.
            Default is (10,10,).
    '''

    def __init__(self, patchSize, exportEmptyPatches=False,
        cropMode='strided',
        stride=None, minBBoxArea=256, minBBoxAreaFrac=0.25,
        cropSize=None, minCropSize=None, maxCropSize=None, forcePatchSizeAspectRatio=True, maintainAspectRatio=True,
        searchStride=(10,10,)):

        self.patchSize = patchSize
        if isinstance(self.patchSize, int):
            self.patchSize = (self.patchSize, self.patchSize,)

        if cropMode == 'strided':
            self.stride = stride
            if self.stride is None:
                self.stride = self.patchSize
            elif isinstance(self.stride, int):
                self.stride = (self.stride, self.stride,)
        
        elif cropMode == 'objectCentered':
            self.cropSize = cropSize
            if self.cropSize is None:
                self.cropSize = self.patchSize
            elif isinstance(self.cropSize, int) or isinstance(self.cropSize, float):
                self.cropSize = (self.cropSize, self.cropSize,)
            
            self.minCropSize = minCropSize
            if self.minCropSize is None:
                self.minCropSize = (0, 0,)
            elif isinstance(self.minCropSize, int):
                self.minCropSize = (self.minCropSize, self.minCropSize,)
            self.maxCropSize = maxCropSize
            if maxCropSize is None:
                self.maxCropSize = (1e9, 1e9,)
            elif isinstance(self.maxCropSize, int):
                self.maxCropSize = (self.maxCropSize, self.maxCropSize,)

        elif cropMode == 'windowCropping':
            self.searchStride = searchStride
            if isinstance(self.searchStride, int):
                self.searchStride = (self.searchStride, self.searchStride,)
        self.cropMode = cropMode
        self.exportEmptyPatches = exportEmptyPatches
        self.minBBoxArea = minBBoxArea
        self.minBBoxAreaFrac = minBBoxAreaFrac
        self.forcePatchSizeAspectRatio = forcePatchSizeAspectRatio
        self.maintainAspectRatio = maintainAspectRatio


    def splitImageIntoPatches(self, image, bboxes, labels, logits):
        sz = image.size

        labels = labels.numpy()
        bboxes = bboxes.numpy()
        logits = logits.numpy()

        # prepare return values
        result = {}

        if len(bboxes):
            # convert to XYWH format
            bboxes[:,2] -= bboxes[:,0]
            bboxes[:,3] -= bboxes[:,1]
            bboxes[:,0] += bboxes[:,2]/2
            bboxes[:,1] += bboxes[:,3]/2


        # create split locations
        if self.cropMode == 'strided':
            # regular grid
            maxX = sz[0] - self.patchSize[0]
            maxY = sz[1] - self.patchSize[1]

            coordsX = np.append(np.arange(0, maxX, self.stride[0], dtype=np.int), maxX)
            coordsY = np.append(np.arange(0, maxY, self.stride[1], dtype=np.int), maxY)

            # expand to all locations
            coordsX, coordsY = np.meshgrid(coordsX, coordsY)

            coordsX = coordsX.ravel()
            coordsY = coordsY.ravel()

            cropSizesX = np.repeat(self.patchSize[0], len(coordsX)).astype(np.int)
            cropSizesY = np.repeat(self.patchSize[1], len(coordsY)).astype(np.int)
        
        elif self.cropMode == 'objectCentered':
            # create positions around bboxes
            if not len(bboxes):
                # nothing in this image; do not export anything
                return result
            
            coordsX = []
            coordsY = []
            cropSizesX = []
            cropSizesY = []

            for l in range(len(bboxes)):

                # calc. patch extents around bbox
                if isinstance(self.cropSize[0], float):
                    widthProj = self.cropSize[0] * bboxes[l,2]
                    heightProj = self.cropSize[1] * bboxes[l,3]
                else:
                    widthProj = self.cropSize[0]
                    heightProj = self.cropSize[1]
                widthX = min(self.maxCropSize[0], max(self.minCropSize[0], widthProj))
                heightX = min(self.maxCropSize[1], max(self.minCropSize[1], heightProj))
                
                if self.forcePatchSizeAspectRatio:
                    scaleFactor = min(self.patchSize[0] / widthX, self.patchSize[1] / heightX)
                    widthX = self.patchSize[0] / scaleFactor
                    heightX = self.patchSize[1] / scaleFactor

                coordsX.append(int(round(bboxes[l,0] - widthX/2)))
                coordsY.append(int(round(bboxes[l,1] - heightX/2)))
                cropSizesX.append(int(round(widthX)))
                cropSizesY.append(int(round(heightX)))

        elif self.cropMode == 'windowCropping':
            # create a minimal number of positions that include as many boxes as possible
            if not len(bboxes):
                # nothing in this image; do not export anything
                return result

            # output
            coordsX = []
            coordsY = []

            if len(bboxes) == 1:
                # no need to cluster
                leftX = int(max(0, min(sz[0] - self.patchSize[0], bboxes[0,0] - self.patchSize[0]/2)))
                topY = int(max(0, min(sz[1] - self.patchSize[1], bboxes[0,1] - self.patchSize[1]/2)))
                coordsX.append(leftX)
                coordsY.append(topY)
            
            else:
                # keep track of boxes already covered
                bboxes_covered = set()
                bboxIndices = np.arange(len(bboxes))

                # identify query order by clustering the coordinates
                numClusters = np.max([2, np.sqrt(len(bboxes))]).astype(np.int)
                kmeans = KMeans(n_clusters=numClusters).fit(bboxes[:,0:2])
                count = np.zeros(numClusters)
                distances = np.zeros(len(bboxes))
                for i in range(numClusters):
                    bboxes_cluster = bboxes[kmeans.labels_==i, 0:2]
                    count[i] = len(bboxes_cluster)
                    distances[kmeans.labels_==i] = np.sum((bboxes_cluster - kmeans.cluster_centers_[i, :]) ** 2, 1)

                # iterate: biggest cluster, lowest distance first
                cluOrder = np.argsort(count)
                cluOrder = cluOrder[::-1]

                for clu in cluOrder:
                    cOrder = np.argsort(distances[kmeans.labels_==clu])
                    candidates = bboxIndices[kmeans.labels_==clu]
                    for can in cOrder:
                        if candidates[can] in bboxes_covered:
                            continue
                        
                        # try patches around candidate
                        nextBBox = bboxes[candidates[can], :]
                        minX = int(max(0, np.ceil(nextBBox[0] + nextBBox[2]/2) - self.patchSize[0] + 1))
                        maxX = int(max(minX, min(sz[0] - self.patchSize[0], np.floor(nextBBox[0] - nextBBox[2]/2))))
                        minY = int(max(0, np.ceil(nextBBox[1] + nextBBox[3]/2) - self.patchSize[1] + 1))
                        maxY = int(max(minY, min(sz[1] - self.patchSize[1], np.floor(nextBBox[1] - nextBBox[3]/2))))
                        bestNumCandidates = 0
                        argMax = (-1, -1,)
                        bestMeanCenterDist = 0      # average distance of included bboxes to patch center

                        searchRangeX = np.arange(minX, maxX, self.searchStride[0])
                        if not len(searchRangeX) or searchRangeX[-1] != maxX:
                            searchRangeX = np.append(searchRangeX, maxX)
                        searchRangeY = np.arange(minY, maxY, self.searchStride[1])
                        if not len(searchRangeY) or searchRangeY[-1] != maxY:
                            searchRangeY = np.append(searchRangeY, maxY)

                        for x in searchRangeX:
                            for y in searchRangeY:
                                numCandidates = 0
                                meanCenterDist = 0
                                for b in bboxIndices:
                                    if b in bboxes_covered:
                                        # we only care about uncovered bboxes
                                        continue

                                    if max(0, bboxes[b, 0] - bboxes[b, 2]/2) >= x and \
                                        min(sz[0]-1, bboxes[b, 0] + bboxes[b, 2]/2) < (x + self.patchSize[0]) and \
                                        max(0, bboxes[b, 1] - bboxes[b, 3]/2) >= y and \
                                        min(sz[1]-1, bboxes[b, 1] + bboxes[b, 3]/2) < (y + self.patchSize[1]):
                                        # bbox inside area; add
                                        numCandidates += 1
                                        meanCenterDist += np.sum((bboxes[b, 0:2] - [x + self.patchSize[0]/2, y + self.patchSize[1]/2]) ** 2)

                                meanCenterDist /= float(numCandidates)
                                if numCandidates > bestNumCandidates:
                                    bestNumCandidates = numCandidates
                                    bestMeanCenterDist = meanCenterDist
                                    argMax = (x, y,)
                                
                                elif numCandidates == bestNumCandidates:
                                    # check if positioning is better
                                    if meanCenterDist < bestMeanCenterDist:
                                        # update with current position
                                        bestMeanCenterDist = meanCenterDist
                                        argMax = (x, y,)

                        if bestNumCandidates == 1:
                            # only one box covered; re-position patch to center it
                            leftX = max(0, min(sz[0] - self.patchSize[0], nextBBox[0] - self.patchSize[0]/2))
                            topY = max(0, min(sz[1] - self.patchSize[1], nextBBox[1] - self.patchSize[1]/2))
                            argMax = (leftX, topY)

                        # mark inclusive bboxes as 'covered'
                        for b in bboxIndices:
                            if b in bboxes_covered:
                                # we only care about uncovered bboxes
                                continue
                            if max(0, bboxes[b, 0] - bboxes[b, 2]/2) >= argMax[0] and \
                                min(sz[0]-1, bboxes[b, 0] + bboxes[b, 2]/2) < (argMax[0] + self.patchSize[0]) and \
                                max(0, bboxes[b, 1] - bboxes[b, 3]/2) >= argMax[1] and \
                                min(sz[1]-1, bboxes[b, 1] + bboxes[b, 3]/2) < (argMax[1] + self.patchSize[1]):
                                bboxes_covered.add(b)
                        
                        # append coordinates
                        coordsX.append(int(argMax[0]))
                        coordsY.append(int(argMax[1]))
            
                # sanity check
                for b in bboxIndices:
                    if b not in bboxes_covered:
                        print('something is wrong')

            cropSizesX = np.repeat(self.patchSize[0], len(coordsX)).astype(np.int)
            cropSizesY = np.repeat(self.patchSize[1], len(coordsY)).astype(np.int)


        if len(bboxes):
            # convert to LTBR format
            bboxes[:,0] -= bboxes[:,2]/2
            bboxes[:,1] -= bboxes[:,3]/2
            bboxes[:,2] += bboxes[:,0]
            bboxes[:,3] += bboxes[:,1]
    
        # iterate, split, reposition and export
        for cIdx in range(len(coordsX)):
            frame = np.array([coordsX[cIdx], coordsY[cIdx], coordsX[cIdx]+cropSizesX[cIdx], coordsY[cIdx]+cropSizesY[cIdx]])
            patch = image.crop(frame)

            # prepare result
            patchKey = '{}_{}_{}_{}'.format(coordsX[cIdx], coordsY[cIdx], cropSizesX[cIdx], cropSizesY[cIdx])
            result[patchKey] = {
                'patch': patch,
                'predictions': []
            }


            # find bounding boxes within frame
            valid = 0
            if len(bboxes):
                valid = (bboxes[:,0] < frame[2]) * \
                        (bboxes[:,1] < frame[3]) * \
                        (bboxes[:,2] > frame[0]) * \
                        (bboxes[:,3] > frame[1])

            if not self.exportEmptyPatches and not np.sum(valid):
                continue
            
            hasBoxes = False
            if np.sum(valid):
                hasBoxes = True

                bboxes_patch = np.copy(bboxes[valid,:])
                labels_patch = labels[valid]
                logits_patch = np.copy(logits[valid,:])

                # adjust bboxes for patch dimensions
                bboxes_patch[:,0] -= frame[0]
                bboxes_patch[:,1] -= frame[1]
                bboxes_patch[:,2] -= frame[0]
                bboxes_patch[:,3] -= frame[1]

                bboxes_patch[:,2] = np.minimum(cropSizesX[cIdx], bboxes_patch[:,2])
                bboxes_patch[:,3] = np.minimum(cropSizesY[cIdx], bboxes_patch[:,3])
                bboxes_patch[:,0] = np.maximum(0, bboxes_patch[:,0])
                bboxes_patch[:,1] = np.maximum(0, bboxes_patch[:,1])
                bboxes_patch[:,2] -=  bboxes_patch[:,0]
                bboxes_patch[:,3] -=  bboxes_patch[:,1]
                bboxes_patch[:,0] += bboxes_patch[:,2]/2
                bboxes_patch[:,1] += bboxes_patch[:,3]/2

                # filter bboxes for size
                origArea = (bboxes[valid,2] - bboxes[valid,0]) * (bboxes[valid,3] - bboxes[valid,1])
                newArea = bboxes_patch[:,2] * bboxes_patch[:,3]
                areaFrac = newArea / origArea

                valid_area = (newArea >= self.minBBoxArea) * (areaFrac >= self.minBBoxAreaFrac)
                bboxes_patch = bboxes_patch[valid_area,:]
                labels_patch = labels_patch[valid_area]
                logits_patch = logits_patch[valid_area,:]

                # rescale image patch if needed
                if self.cropMode == 'objectCentered':
                    sz_orig = [float(cropSizesX[cIdx]), float(cropSizesY[cIdx])]
                    sz_new = [float(self.patchSize[0]), float(self.patchSize[1])]
                    scaleFactor = [sz_new[0] / sz_orig[0], sz_new[1] / sz_orig[1]]
                    if self.maintainAspectRatio:
                        scaleFactor = min(scaleFactor[0], scaleFactor[1])
                        scaleFactor = [scaleFactor, scaleFactor]
                    sz_new = [scaleFactor[0] * sz_orig[0], scaleFactor[1] * sz_orig[1]]

                    patch = patch.resize([int(s) for s in sz_new], Image.BILINEAR)
                    
                    if len(bboxes_patch):
                        # adjust origin
                        bboxes_patch[:,0] = (bboxes_patch[:,0] - sz_orig[0]/2) * (scaleFactor[0]) + sz_new[0]/2
                        bboxes_patch[:,1] = (bboxes_patch[:,1] - sz_orig[1]/2) * (scaleFactor[1]) + sz_new[1]/2

                        # adjust width and height
                        bboxes_patch[:,2] *= scaleFactor[0]
                        bboxes_patch[:,3] *= scaleFactor[1]

                        # convert bboxes back to YOLO format
                        bboxes_patch[:,0] /= float(sz_new[0])
                        bboxes_patch[:,1] /= float(sz_new[1])
                        bboxes_patch[:,2] /= float(sz_new[0])
                        bboxes_patch[:,3] /= float(sz_new[1])
                    else:
                        hasBoxes = False
                else:
                    # convert bboxes back to YOLO format
                    if np.sum(valid_area):
                        bboxes_patch[:,0] /= float(self.patchSize[0])
                        bboxes_patch[:,1] /= float(self.patchSize[1])
                        bboxes_patch[:,2] /= float(self.patchSize[0])
                        bboxes_patch[:,3] /= float(self.patchSize[1])
                    else:
                        hasBoxes = False

            if not self.exportEmptyPatches and not hasBoxes:
                continue


            # append results
            if self.exportEmptyPatches or (hasBoxes and len(labels_patch)):
                result[patchKey] = {}
                result[patchKey]['patch'] = patch
                result[patchKey]['predictions'] = []
                if hasBoxes and len(labels_patch):
                    for l in range(len(labels_patch)):
                        result[patchKey]['predictions'].append({
                            'x': bboxes_patch[l,0].tolist(),
                            'y': bboxes_patch[l,1].tolist(),
                            'width': bboxes_patch[l,2].tolist(),
                            'height': bboxes_patch[l,3].tolist(),
                            'label': labels_patch[l],
                            'logits': logits_patch[l,:].tolist(),
                            'confidence': np.max(logits_patch[l,:]).tolist()
                        })

        return result