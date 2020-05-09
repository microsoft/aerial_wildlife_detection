'''
    Transformations compatible with image data
    and segmentation masks simultaneously.

    2020 Benjamin Kellenberger
'''

import importlib
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


''' Functionals '''
def _horizontalFlip(img, segMask=None):
    img = F.hflip(img)
    if segMask is not None:
        segMask = F.hflip(segMask)
    return img, segMask


def _verticalFlip(img, segMask=None):
    img = F.vflip(img)
    if segMask is not None:
        segMask = F.vflip(segMask)
    return img, segMask


def _clipPatch(img_in, segMask, patchSize, jitter=(0,0,), limitBorders=False):
    '''
        Clips a patch of size "patchSize" from both "img_in" and "segMask"
        at the same location. "img_in" and "segMask" can either be PIL images
        or else Torch tensors.

        "jitter" is a scalar or tuple of scalars defining the maximum pixel
        values that are randomly added or subtracted to the X and Y coordinates
        of the patch location to maximize variability.

        If "limitBorders" is set to True, the clipped patch will not exceed the
        image boundaries.
    '''

    # setup
    if isinstance(patchSize, int) or isinstance(patchSize, float):
        patchSize = tuple((patchSize, patchSize))
    patchSize = tuple((int(patchSize[0]), int(patchSize[1])))

    if jitter is None:
        jitter = 0.0
    elif isinstance(jitter, int) or isinstance(jitter, float):
        jitter = tuple((jitter, jitter))
    jitter = tuple((float(jitter[0]), float(jitter[1])))

    if isinstance(img_in, torch.Tensor):
        sz = tuple((img_in.size(2), img_in.size(1)))
        img = img_in.clone()
    else:
        sz = tuple((img_in.size[0], img_in.size[1]))
        img = img_in.copy()

    # clip
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
        img_clip = torch.zeros((img.size(0), int(patchSize[1]), int(patchSize[0]),), dtype=img.dtype, device=img.device)
        img_clip[:,0:coordinates_input[3],0:coordinates_input[2]] = img[:,coordinates_input[1]:(coordinates_input[1]+coordinates_input[3]), coordinates_input[0]:(coordinates_input[0]+coordinates_input[2])]
    else:
        img_clip = img.crop((coordinates[0], coordinates[1], coordinates[0]+coordinates[2], coordinates[1]+coordinates[3],))

    if segMask is not None:
        if isinstance(segMask, torch.Tensor):
            coordinates_input = tuple((coordinates[0], coordinates[1], min((sz[0]-coordinates[0]), coordinates[2]), min((sz[1]-coordinates[1]), coordinates[3])))
            segMask_clip = torch.zeros((segMask.size(0), int(patchSize[1]), int(patchSize[0]),), dtype=segMask.dtype, device=segMask.device)
            segMask_clip[:,0:coordinates_input[3],0:coordinates_input[2]] = segMask[:,coordinates_input[1]:(coordinates_input[1]+coordinates_input[3]), coordinates_input[0]:(coordinates_input[0]+coordinates_input[2])]
        else:
            segMask_clip = segMask.crop((coordinates[0], coordinates[1], coordinates[0]+coordinates[2], coordinates[1]+coordinates[3],))
    else:
        segMask_clip = None

    return img_clip, segMask_clip, coordinates


''' Class definitions '''
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, img, segMask=None):
        for t in self.transforms:
            img, segMask = t(img, segMask)
        return img, segMask



class DefaultTransform(object):

    def __init__(self, transform, transform_kwargs=None):
        self.transform = transform

        # load transform if string
        if isinstance(self.transform, str):
            idx = self.transform.rfind('.')
            classPath, executableName = self.transform[0:idx], self.transform[idx+1:]
            execFile = importlib.import_module(classPath)
            self.transform = getattr(execFile, executableName)(**transform_kwargs)


    def __call__(self, img, segMask=None):
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = self.transform(img[i])
        else:
            img = self.transform(img)
        return img, segMask



class JointTransform(object):

    def __init__(self, transform, transform_kwargs=None):
        self.transform = transform

        # load transform if string
        if isinstance(self.transform, str):
            idx = self.transform.rfind('.')
            classPath, executableName = self.transform[0:idx], self.transform[idx+1:]
            execFile = importlib.import_module(classPath)
            self.transform = getattr(execFile, executableName)(**transform_kwargs)


    def __call__(self, img, segMask=None):
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = self.transform(img[i])
                if segMask[i] is not None:
                    segMask[i] = self.transform(segMask[i])
        else:
            img = self.transform(img)
            if segMask is not None:
                segMask = self.transform(segMask[i])
        return img, segMask



class RandomClip(object):

    def __init__(self, patchSize, jitter, limitBorders, numClips=1):
        self.patchSize = patchSize
        self.jitter = jitter
        self.limitBorders = limitBorders
        self.numClips = numClips
    

    def __call__(self, img, segMask=None):
        if self.numClips>1:
            patch = []
            segMask_out = []
            for _ in range(self.numClips):
                patch_n, segMask_out_n, _ = _clipPatch(img, segMask, self.patchSize, self.jitter, self.limitBorders)
                patch.append(patch_n)
                segMask_out.append(segMask_out_n)
        else:
            patch, segMask_out, _ = _clipPatch(img, segMask, self.patchSize, self.jitter, self.limitBorders)
        return patch, segMask_out



class RandomSizedClip(object):

    def __init__(self, patchSizeMin, patchSizeMax, jitter, limitBorders, numClips=1):
        self.patchSizeMin = patchSizeMin
        if isinstance(self.patchSizeMin, int) or isinstance(self.patchSizeMin, float):
            self.patchSizeMin = (self.patchSizeMin, self.patchSizeMin,)
        self.patchSizeMax = patchSizeMax
        if isinstance(self.patchSizeMax, int) or isinstance(self.patchSizeMax, float):
            self.patchSizeMax = (self.patchSizeMax, self.patchSizeMax,)
        self.jitter = jitter
        self.limitBorders = limitBorders
        self.numClips = numClips
    

    def __call__(self, img, segMask=None):
        if self.numClips>1:
            patch = []
            segMasks_out = []
            for n in range(self.numClips):
                patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
                jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
                patch_n, segMask_out_n, _ = _clipPatch(img, segMask, patchSize, jitter, self.limitBorders)
                patch.append(patch_n)
                segMasks_out.append(segMask_out_n)
        else:
            patchSize = (random.randint(self.patchSizeMin[0], self.patchSizeMax[0]), random.randint(self.patchSizeMin[1], self.patchSizeMax[1]),)
            jitter = (min(patchSize[0]/2, self.jitter[0]), min(patchSize[1]/2, self.jitter[1]),)
            patch, segMasks_out, _ = _clipPatch(img, segMask, patchSize, jitter, self.limitBorders)
        return patch, segMasks_out



class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size,int):
            size = tuple((size,size))
        self.size = (size[1], size[0],)
        self.interpolation = interpolation
        if isinstance(self.interpolation, str):
            self.interpolation = self.interpolation.upper()
            if self.interpolation == 'NEAREST':
                self.interpolation = Image.NEAREST
            elif self.interpolation == 'BILINEAR':
                self.interpolation = Image.BILINEAR
            elif self.interpolation == 'BICUBIC':
                self.interpolation = Image.BILINEAR
            elif hasattr(Image, self.interpolation):
                self.interpolation = getattr(Image, self.interpolation)
            else:
                # unparsable; issue warning
                print(f'WARNING: interpolation mode "{self.interpolation}" not understood; set to bilinear instead.')
                self.interpolation = Image.BILINEAR
    

    def __call__(self, img, segMask=None):
        img = F.resize(img, self.size, self.interpolation)
        if segMask is not None:
            segMask = F.resize(img, self.size, Image.NEAREST)
        
        return img, segMask



class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, segMask=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p:
                    img[i], segMask[i] = _horizontalFlip(img[i], segMask[i])
        else:
            if random.random() < self.p:
                img, segMask = _horizontalFlip(img, segMask)
        return img, segMask



class RandomFlip(object):
    
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, img, segMask=None):
        if isinstance(img, list):
            for i in range(len(img)):
                if random.random() < self.p_h:
                    img[i], segMask[i] = _horizontalFlip(img[i], segMask[i])
                
                if random.random() < self.p_v:
                    img[i], segMask[i] = _verticalFlip(img[i], segMask[i])
        else:
            if random.random() < self.p_h:
                img, segMask = _horizontalFlip(img, segMask)
            
            if random.random() < self.p_v:
                img, segMask = _verticalFlip(img, segMask)
        return img, segMask



class RandomRot90(object):

    def __init__(self, stops, p):
        self.stops = min(3,abs(stops))
        self.p = p


    def __call__(self, img, segMask=None):
        if random.random() < self.p:
            numStops = random.randint(-self.stops,self.stops)
            if numStops==0:
                return img, segMask

            angle = numStops*90
            img = F.rotate(img, angle, resample=False, expand=False)
            segMask = F.rotate(segMask, angle, resample=False, expand=False)
        
        return img, segMask