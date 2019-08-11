'''
    Transformations that work on image data and points simultaneously.

    2019 Benjamin Kellenberger
'''

import random
import numpy as np      #TODO: replace np.random.choice with regular random.choice
import torch
import torchvision.transforms.functional as F
from PIL import Image


""" Functionals """
def _horizontalFlip(img, points=None, labels=None):
    img = F.hflip(img)
    if points is not None and len(labels):
        points[:,0] = img.size[0] - points[:,0]
    return img, points, labels


def _verticalFlip(img, points=None, labels=None):
    img = F.vflip(img)
    if points is not None and len(labels):
        points[:,1] = img.size[1] - points[:,1]
    return img, points, labels


def _pointTranslate(points, sz_orig, sz_new):
    sz_orig = [float(s) for s in sz_orig]
    sz_new = [float(s) for s in sz_new]
    points[:,0] *= sz_new[0] / sz_orig[0]
    points[:,1] *= sz_new[1] / sz_orig[1]
    return points


def _clipPatch(img_in, points_in, labels_in, patchSize, jitter=(0,0,), limitBorders=False, objectProbability=None):
    """
        Clips a patch of size 'patchSize' from 'img_in' (either a PIL image or a torch tensor).
        Also returns a subset of 'points' and associated 'labels' that are inside the patch;
        points get adjusted in coordinates to fit the patch.
        The format of the points is (X, Y) in absolute pixel values.

        'jitter' is a scalar or tuple of scalars defining maximum pixel values that are randomly
        added or subtracted to the X and Y coordinates of the patch to maximize variability.

        If 'limitBorders' is set to True, the clipped patch will not exceed the image boundaries.

        If 'objectProbability' is set to a scalar in [0, 1], the patch will be clipped from one of
        the points (if available) drawn at random, under the condition that a uniform, random value
        is <= 'objectProbability'.
        Otherwise the patch will be clipped completely at random.

        Inputs:
        - img_in:               PIL Image object or torch tensor [BxW0xH0]
        - points_in:            torch tensor [N0x2]
        - labels_in:            torch tensor [N0]
        - patchSize:            int or tuple (WxH)
        - jitter:               int or tuple (WxH)
        - limitBorders:         bool
        - objectProbability:    None or scalar in [0, 1]

        Returns:
        - patch:                PIL Image object or torch tensor [BxWxH], depending on format of 'img'
        - points:               torch tensor [Nx2] (subset; copy of original points)
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

    if points_in is None:
        numAnnotations = 0
        points = None
        labels = None
    else:
        points = points_in.clone()
        labels = labels_in.clone()
        if points.dim()==1:
            points = points.unsqueeze(0)
        if labels.dim()==0:
            labels = labels.unsqueeze(0)
        numAnnotations = labels.size(0)

    if isinstance(img_in, torch.Tensor):
        sz = tuple((img_in.size(2), img_in.size(1)))
        img = img_in.clone()
    else:
        sz = tuple((img_in.size[0], img_in.size[1]))
        img = img_in.copy()


    # clip
    if objectProbability is not None and numAnnotations and random.random() <= objectProbability:
        # clip patch from around points
        idx = np.random.choice(numAnnotations, 1)
        baseCoords = points[idx, 0:2].squeeze()
        baseCoords[0] -= patchSize[0]/2.0
        baseCoords[1] -= patchSize[1]/2.0
    
    else:
        # clip at random
        baseCoords = torch.tensor([np.random.choice(sz[0], 1), np.random.choice(sz[1], 1)], dtype=torch.float32).squeeze()
    
    # jitter
    jitterAmount = (2*jitter[0]*(random.random()-0.5), 2*jitter[1]*(random.random()-0.5),)
    baseCoords[0] += jitterAmount[0]
    baseCoords[1] += jitterAmount[1]

    # sanity check
    baseCoords = torch.ceil(baseCoords).int()
    if limitBorders:
        baseCoords[0] = max(0, baseCoords[0])
        baseCoords[1] = max(0, baseCoords[1])
        baseCoords[0] = min(sz[0]-patchSize[0], baseCoords[0])
        baseCoords[1] = min((sz[1]-patchSize[1]), baseCoords[1])

    # assemble
    coordinates = tuple((baseCoords[0].item(), baseCoords[1].item(), patchSize[0], patchSize[1]))


    # do the clipping
    if isinstance(img, torch.Tensor):
        coordinates_input = tuple((coordinates[0], coordinates[1], min((sz[0]-coordinates[0]), coordinates[2]), min((sz[1]-coordinates[1]), coordinates[3])))
        patch = torch.zeros((img.size(0), int(patchSize[1]), int(patchSize[0]),), dtype=img.dtype, device=img.device)
        patch[:,0:coordinates_input[3],0:coordinates_input[2]] = img[:,coordinates_input[1]:(coordinates_input[1]+coordinates_input[3]), coordinates_input[0]:(coordinates_input[0]+coordinates_input[2])]
    else:
        patch = img.crop((coordinates[0], coordinates[1], coordinates[0]+coordinates[2], coordinates[1]+coordinates[3],))

    
    # select and translate points
    if numAnnotations:
        coords_f = [float(c) for c in coordinates]
        valid = (points[:,0] >= coords_f[0]) * (points[:,1] >= coords_f[1]) * \
                (points[:,0] < (coords_f[0]+coords_f[2])) * (points[:,1] < (coords_f[1]+coords_f[3]))
        
        points = points[valid,:]
        labels = labels[valid]

        points[:,0] -= coords_f[0]
        points[:,1] -= coords_f[1]

    return patch, points, labels, coordinates


""" Class definitions """
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    

    def __call__(self, img, points=None, labels=None):
        for t in self.transforms:
            img, points, labels = t(img, points, labels)
        return img, points, labels



class DefaultTransform(object):

    def __init__(self, transform):
        self.transform = transform


    def __call__(self, img, points=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = self.transform(img[i])
        else:
            img = self.transform(img)
        return img, points, labels




class RandomClip(object):

    def __init__(self, patchSize, jitter, limitBorders, objectProbability, numClips=1):
        self.patchSize = patchSize
        self.jitter = jitter
        self.limitBorders = limitBorders
        self.objectProbability = objectProbability
        self.numClips = numClips
    

    def __call__(self, img, points=None, labels=None):
        if self.numClips>1:
            patch = []
            points_out = []
            labels_out = []
            for n in range(self.numClips):
                patch_n, points_out_n, labels_out_n, _ = _clipPatch(img, points, labels, self.patchSize, self.jitter, self.limitBorders, self.objectProbability)
                patch.append(patch_n)
                points_out.append(points_out_n)
                labels_out.append(labels_out_n)
        else:
            patch, points_out, labels_out, _ = _clipPatch(img, points, labels, self.patchSize, self.jitter, self.limitBorders, self.objectProbability)
        return patch, points_out, labels_out





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
    

    def __call__(self, img, points=None, labels=None):
        if self.numClips>1:
            patch = []
            points_out = []
            labels_out = []
            for n in range(self.numClips):
                patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
                jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
                patch_n, points_out_n, labels_out_n, _ = _clipPatch(img, points, labels, patchSize, jitter, self.limitBorders, self.objectProbability)
                patch.append(patch_n)
                points_out.append(points_out_n)
                labels_out.append(labels_out_n)
        else:
            patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
            jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
            patch, points_out, labels_out, _ = _clipPatch(img, points, labels, patchSize, jitter, self.limitBorders, self.objectProbability)
        return patch, points_out, labels_out





class Resize(object):
    """
        Works on a PIL image and points with format (X, Y) and absolute pixel values.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size,int):
            size = tuple((size,size))
        self.size = (size[1], size[0],)
        self.interpolation = interpolation
    

    def __call__(self, img, points=None, labels=None):
        sz_orig = img.size
        img = F.resize(img, self.size, self.interpolation)
        sz_new = img.size

        if points is not None and len(points) > 0:
            points = _pointTranslate(points, sz_orig, sz_new)

        return img, points, labels




class RandomHorizontalFlip(object):

    def __init__(self,p=0.5):
        self.p = p
    
    def __call__(self, img, points=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p:
                    img[i], points[i], labels[i] = _horizontalFlip(img[i], points[i], labels[i])
        else:
            if random.random() < self.p:
                img, points, labels = _horizontalFlip(img, points, labels)
        return img, points, labels



class RandomFlip(object):
    
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v


    def __call__(self, img, points=None, labels=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p_h:
                    img[i], points[i], labels[i] = _horizontalFlip(img[i], points[i], labels[i])
                
                if random.random() < self.p_v:
                    img[i], points[i], labels[i] = _verticalFlip(img[i], points[i], labels[i])
        else:
            if random.random() < self.p_h:
                img, points, labels = _horizontalFlip(img, points, labels)
            
            if random.random() < self.p_v:
                img, points, labels = _verticalFlip(img, points, labels)
        return img, points, labels



class RandomRot90(object):

    def __init__(self, stops, p):
        self.stops = min(3,abs(stops))
        self.p = p


    def __call__(self, img, points=None, labels=None):
        if random.random() < self.p:
            numStops = random.randint(-self.stops,self.stops)
            if numStops==0:
                return img, points, labels

            angle = numStops*90
            img = F.rotate(img, angle, resample=False, expand=False)

            #TODO: warp points correctly
            points_copy = points.clone()
            #continue...
        
        return img, points, labels



