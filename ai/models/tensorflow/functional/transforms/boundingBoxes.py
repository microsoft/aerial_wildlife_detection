'''
    Transformations that work on image data and bounding boxes simultaneously.

    2019 Benjamin Kellenberger
'''

import random
import numpy as np      #TODO: replace np.random.choice with regular random.choice
import torchvision.transforms.functional as F
from PIL import Image


""" Functionals """
def _horizontalFlip(img, bboxes=None, labels=None):
    img = F.hflip(img)
    if bboxes is not None and len(labels):
        bboxes[:,0] = img.size[0] - (bboxes[:,0] + bboxes[:,2])
    return img, bboxes, labels


def _verticalFlip(img, bboxes=None, labels=None):
    img = F.vflip(img)
    if bboxes is not None and len(labels):
        bboxes[:,1] = img.size[1] - (bboxes[:,1] + bboxes[:,3])
    return img, bboxes, labels


def _bboxResize(bboxes, sz_orig, sz_new):
    sz_orig = [float(s) for s in sz_orig]
    sz_new = [float(s) for s in sz_new]

    # adjust origin
    bboxes[:,0] = (bboxes[:,0] - sz_orig[0]/2) * (sz_new[0] / sz_orig[0]) + sz_new[0]/2
    bboxes[:,1] = (bboxes[:,1] - sz_orig[1]/2) * (sz_new[1] / sz_orig[1]) + sz_new[1]/2

    # adjust width and height
    bboxes[:,2] *= sz_new[0] / sz_orig[0]
    bboxes[:,3] *= sz_new[1] / sz_orig[1]
    return bboxes


def _clipPatch(img_in, bboxes_in, labels_in, patchSize, jitter=(0,0,), limitBorders=False, objectProbability=None):
    """
        Clips a patch of size 'patchSize' from 'img_in' (either a PIL image or a torch tensor).
        Also returns a subset of 'bboxes' and associated 'labels' that are inside the patch;
        bboxes get adjusted in coordinates to fit the patch.
        The format of the bboxes is (X, Y, W, H) in absolute pixel values, with X and Y denoting
        the top left corner of the respective bounding box.

        'jitter' is a scalar or tuple of scalars defining maximum pixel values that are randomly
        added or subtracted to the X and Y coordinates of the patch to maximize variability.

        If 'limitBorders' is set to True, the clipped patch will not exceed the image boundaries.

        If 'objectProbability' is set to a scalar in [0, 1], the patch will be clipped from one of
        the bboxes (if available) drawn at random, under the condition that a uniform, random value
        is <= 'objectProbability'.
        Otherwise the patch will be clipped completely at random.

        Inputs:
        - img_in:               PIL Image object or torch tensor [BxW0xH0]
        - bboxes_in:            torch tensor [N0x4]
        - labels_in:            torch tensor [N0]
        - patchSize:            int or tuple (WxH)
        - jitter:               int or tuple (WxH)
        - limitBorders:         bool
        - objectProbability:    None or scalar in [0, 1]

        Returns:
        - patch:                PIL Image object or torch tensor [BxWxH], depending on format of 'img'
        - bboxes:               torch tensor [Nx4] (subset; copy of original bboxes)
        - labels:               torch tensor [N] (ditto)
        - coordinates:          tuple (X, Y, W, H coordinates of clipped patch)
    """


    # setup
    if isinstance(patchSize, int) or isinstance(patchSize, float):
        patchSize = tuple((patchSize, patchSize))
    patchSize = tuple((int(patchSize[0]), int(patchSize[1])))

    if jitter is None:
        jitter = 0.0
    elif isinstance(jitter, int) or isinstance(jitter, float):
        jitter = tuple((jitter, jitter))
    jitter = tuple((float(jitter[0]), float(jitter[1])))

    if bboxes_in is None:
        numAnnotations = 0
        bboxes = bboxes_in
        labels = None
    else:

        bboxes = bboxes_in.copy()
        labels = np.array(labels_in)
        numAnnotations = len(labels)

    sz = tuple((img_in.size[0], img_in.size[1]))
    img = img_in.copy()


    # clip
    if objectProbability is not None and numAnnotations and random.random() <= objectProbability:
        # clip patch from around bounding box
        idx = np.random.choice(numAnnotations, 1)
        baseCoords = bboxes[idx, 0:2].squeeze() + bboxes[idx, 2:].squeeze()/2.0
        baseCoords[0] -= patchSize[0]/2.0
        baseCoords[1] -= patchSize[1]/2.0
    
    else:
        # clip at random
        baseCoords = np.array([np.random.choice(sz[0], 1), np.random.choice(sz[1], 1)], dtype=np.float32).squeeze()
    
    # jitter
    jitterAmount = (2*jitter[0]*(random.random()-0.5), 2*jitter[1]*(random.random()-0.5),)
    baseCoords[0] += jitterAmount[0]
    baseCoords[1] += jitterAmount[1]

    # sanity check
    baseCoords = np.ceil(baseCoords).astype(int)
    if limitBorders:
        baseCoords[0] = max(0, baseCoords[0])
        baseCoords[1] = max(0, baseCoords[1])
        baseCoords[0] = min(sz[0]-patchSize[0], baseCoords[0])
        baseCoords[1] = min(sz[1]-patchSize[1], baseCoords[1])

    # assemble
    coordinates = tuple((baseCoords[0], baseCoords[1], patchSize[0], patchSize[1]))
    


    # do the clipping
    patch = img.crop((coordinates[0], coordinates[1], coordinates[0]+coordinates[2], coordinates[1]+coordinates[3],))

    
    # limit and translate bounding boxes
    if numAnnotations:
        coords_f = [float(c) for c in coordinates]
        valid = ((bboxes[:,0]+bboxes[:,2]) >= coords_f[0]) * ((bboxes[:,1]+bboxes[:,3]) >= coords_f[1]) * \
                (bboxes[:,0] < (coords_f[0]+coords_f[2])) * (bboxes[:,1] < (coords_f[1]+coords_f[3]))
        
        bboxes = bboxes[valid,:]
        labels = labels[valid].tolist()

        # translate and limit to image borders
        bboxes[:,0] -= coords_f[0]
        bboxes[:,1] -= coords_f[1]
        diffX = np.zeros_like(bboxes[:,0]) - bboxes[:,0]
        diffY = np.zeros_like(bboxes[:,1]) - bboxes[:,1]
        bboxes[diffX>0,2] -= diffX[diffX>0]
        bboxes[diffY>0,3] -= diffY[diffY>0]
        bboxes[:,0] = np.maximum(np.zeros_like(bboxes[:,0]), bboxes[:,0])
        bboxes[:,1] = np.maximum(np.zeros_like(bboxes[:,1]), bboxes[:,1])
        bboxes[:,2] = np.minimum((patchSize[0]*np.ones_like(bboxes[:,2]) - bboxes[:,0]), bboxes[:,2])
        bboxes[:,3] = np.minimum((patchSize[1]*np.ones_like(bboxes[:,3]) - bboxes[:,1]), bboxes[:,3])

    return patch, bboxes, labels, coordinates


""" Class definitions """
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    

    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels



class DefaultTransform(object):

    def __init__(self, transform):
        self.transform = transform


    def __call__(self, img, bboxes=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = self.transform(img[i])
        else:
            img = self.transform(img)
        return img, bboxes, labels




class RandomClip(object):

    def __init__(self, patchSize, jitter, limitBorders, objectProbability, numClips=1):
        self.patchSize = patchSize
        self.jitter = jitter
        self.limitBorders = limitBorders
        self.objectProbability = objectProbability
        self.numClips = numClips
    

    def __call__(self, img, bboxes=None, labels=None):

        if self.numClips>1:
            patch = []
            bboxes_out = []
            labels_out = []
            for n in range(self.numClips):
                patch_n, bboxes_out_n, labels_out_n, _ = _clipPatch(img, bboxes, labels, self.patchSize, self.jitter, self.limitBorders, self.objectProbability)
                patch.append(patch_n)
                bboxes_out.append(bboxes_out_n)
                labels_out.append(labels_out_n)
        else:
            patch, bboxes_out, labels_out, _ = _clipPatch(img, bboxes, labels, self.patchSize, self.jitter, self.limitBorders, self.objectProbability)
        return patch, bboxes_out, labels_out





class RandomSizedClip(object):

    def __init__(self, patchSizeMin, patchSizeMax, jitter, limitBorders, objectProbability, numClips=1):
        self.patchSizeMin = patchSizeMin
        if isinstance(self.patchSizeMin, int) or isinstance(self.patchSizeMin, float):
            self.patchSizeMin = (self.patchSizeMin, self.patchSizeMin,)
        self.patchSizeMax = patchSizeMax
        if isinstance(self.patchSizeMax, int) or isinstance(self.patchSizeMax, float):
            self.patchSizeMax = (self.patchSizeMax, self.patchSizeMax,)
        self.jitter = jitter
        self.limitBorders = limitBorders
        self.objectProbability = objectProbability
        self.numClips = numClips
    

    def __call__(self, img, bboxes=None, labels=None):
        if self.numClips>1:
            patch = []
            bboxes_out = []
            labels_out = []
            for n in range(self.numClips):
                patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
                jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
                patch_n, bboxes_out_n, labels_out_n, _ = _clipPatch(img, bboxes, labels, patchSize, jitter, self.limitBorders, self.objectProbability)
                patch.append(patch_n)
                bboxes_out.append(bboxes_out_n)
                labels_out.append(labels_out_n)
        else:
            patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
            jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
            patch, bboxes_out, labels_out, _ = _clipPatch(img, bboxes, labels, patchSize, jitter, self.limitBorders, self.objectProbability)
        return patch, bboxes_out, labels_out





class Resize(object):
    """
        Works on a PIL image and bboxes with format (X, Y, W, H) and absolute pixel values.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size,int):
            size = tuple((size,size))
        self.size = (size[1], size[0],)
        self.interpolation = interpolation
    

    def __call__(self, img, bboxes=None, labels=None):
        sz_orig = img.size
        img = img.resize(self.size, self.interpolation)
        sz_new = img.size

        if bboxes is not None and len(bboxes) > 0:
            bboxes = _bboxResize(bboxes, sz_orig, sz_new)

        return img, bboxes, labels




class RandomHorizontalFlip(object):

    def __init__(self,p=0.5):
        self.p = p
    
    def __call__(self, img, bboxes=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p:
                    img[i], bboxes[i], labels[i] = _horizontalFlip(img[i], bboxes[i], labels[i])
        else:
            if random.random() < self.p:
                img, bboxes, labels = _horizontalFlip(img, bboxes, labels)
        return img, bboxes, labels



class RandomFlip(object):
    
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v


    def __call__(self, img, bboxes=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p_h:
                    img[i], bboxes[i], labels[i] = _horizontalFlip(img[i], bboxes[i], labels[i])
                
                if random.random() < self.p_v:
                    img[i], bboxes[i], labels[i] = _verticalFlip(img[i], bboxes[i], labels[i])
        else:
            if random.random() < self.p_h:
                img, bboxes, labels = _horizontalFlip(img, bboxes, labels)
            
            if random.random() < self.p_v:
                img, bboxes, labels = _verticalFlip(img, bboxes, labels)
        return img, bboxes, labels



class RandomRot90(object):

    def __init__(self, stops, p):
        self.stops = min(3,abs(stops))
        self.p = p


    def __call__(self, img, bboxes=None, labels=None):
        if random.random() < self.p:
            numStops = random.randint(-self.stops,self.stops)
            if numStops==0:
                return img, bboxes, labels

            angle = numStops*90
            img = F.rotate(img, angle, resample=False, expand=False)

            #TODO: warp bboxes correctly
            bboxes_copy = bboxes.clone()
            #bboxes[:,0] 
        
        return img, bboxes, labels



