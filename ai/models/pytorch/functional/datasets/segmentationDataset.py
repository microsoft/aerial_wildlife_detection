'''
    PyTorch dataset wrapper for segmentation masks.

    2020 Benjamin Kellenberger
'''

import numpy as np
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    '''
        #TODO: update description
        PyTorch-conform wrapper for a dataset containing bounding boxes.
        Inputs:
        - data:     A dict with the following entries:
                    - labelClasses: dict with { <labelclass UUID> : { 'index' (labelClass index for this CNN) }}
                    - images: dict with
                              { <image UUID> : { 'annotations' : { <annotation UUID> : { 'segmentationmask', 'width', 'height' }}}}
                    - 'fVec': optional, contains feature vector bytes for image
        - fileServer: Instance that implements a 'getFile' function to load images
        - labelclassMap: a dictionary/LUT with mapping: key = label class UUID, value = index (number) according
                         to the model.
        - transform: Instance of classes defined in 'ai.models.pytorch.functional.transforms.segmentationMasks'. May be None for no transformation at all.

        The '__getitem__' function returns the data entry at given index as a tuple with the following contents:
        - img: the loaded and transformed (if specified) image.
        - segmentationMask: the loaded and transformed (if specified) segmentation mask.
        - imageID: str, filename of the image loaded
    '''
    def __init__(self, data, fileServer, labelclassMap, transform=None):
        super(SegmentationDataset, self).__init__()
        self.data = data
        self.fileServer = fileServer
        self.labelclassMap = labelclassMap
        self.transform = transform
        self.imageOrder = list(self.data['images'].keys())


    def __len__(self):
        return len(self.imageOrder)

    
    def __getitem__(self, idx):
        imageID = self.imageOrder[idx]
        dataDesc = self.data['images'][imageID]

        #TODO
        from celery.contrib import rdb
        rdb.set_trace()

        # load image
        imagePath = dataDesc['filename']
        try:
            img = Image.open(BytesIO(self.fileServer.getFile(imagePath))).convert('RGB')
        except:
            print(f'WARNING: Image "{imagePath}"" is corrupt and could not be loaded.')
            img = None
        
        # load and decode segmentation mask
        annotationID = dataDesc['annotations']  #TODO: empty images are provided too, fix first
        try:
            width = dataDesc['width']       #TODO
            height = dataDesc['height']
            raster = np.frombuffer(base64.b64decode(dataDesc['segmentationmask']), dtype=np.uint8)
            raster = np.reshape(raster, (height,width,))
            segmentationMask = Image.fromarray(raster)
        except:
            print(f'WARNING: Segmentation mask for image "{imagePath}" could not be loaded or decoded.')
            segmentationMask = None
        
        if self.transform is not None and img is not None and segmentationMask is not None:
            img, segmentationMask = self.transform(img, segmentationMask)
        
        return img, segmentationMask, imageID
