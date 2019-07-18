'''
    PyTorch dataset wrapper, optimized for the AL platform.

    2019 Benjamin Kellenberger
'''

from io import BytesIO
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class BoundingBoxDataset(Dataset):
    '''
        PyTorch-conform wrapper for a dataset containing bounding boxes.
        Inputs:
        - data:     A dict with the following entries:
                    - labelClasses: dict with { <labelclass UUID> : { 'index' (optional: labelClass index for this CNN) }}
                                    If no index is provided for each class, it will be generated in key order.
                    - images: dict with
                              { <image UUID> : { 'annotations' : { <annotation UUID> : { 'x', 'y', 'width', 'height', 'label' (label UUID) }}}}
                    - 'fVec': optional, contains feature vector bytes for image
        - fileServer: Instance that implements a 'getFile' function to load images
        - labelClassMap: a dictionary/LUT with mapping: key = label class UUID, value = index (number) according
                         to the model.
        - targetFormat: str for output bounding box format, either 'xywh' (xy = center coordinates, wh = width & height)
                        or 'xyxy' (top left and bottom right coordinates)
        - transform: Instance of classes defined in 'ai.functional.pytorch._util.bboxTransforms'. May be None for no transformation at all.
        - ignoreUnsure: if True, all annotations with flag 'unsure' will get a label of -1 (i.e., 'ignore')

        The '__getitem__' function returns the data entry at given index as a tuple with the following contents:
        - img: the loaded and transformed (if specified) image. Note: if 'loadImage' is set to False, 'img' will be None.
        - boundingBoxes: transformed bounding boxes for the image.
        - labels: labels for each bounding box according to the dataset's 'labelClassMap' LUT (i.e., the labelClass indices).
        - fVec: a torch tensor of feature vectors (if available; else None)
        - imageID: str, filename of the image loaded
    '''
    def __init__(self, data, fileServer, labelClassMap, targetFormat='xywh', transform=None, ignoreUnsure=False):
        super(BoundingBoxDataset, self).__init__()
        self.fileServer = fileServer
        self.labelClassMap = labelClassMap
        self.targetFormat = targetFormat
        self.transform = transform
        self.ignoreUnsure = ignoreUnsure
        self.__parse_data(data)

    
    def __parse_data(self, data):
        
        # create inverse label class map as well
        self.labelClassMap_inv = {}
        for key in self.labelClassMap.keys():
            val = self.labelClassMap[key]
            self.labelClassMap_inv[val] = key
        
        # parse images
        self.data = []
        for key in data['images']:
            nextMeta = data['images'][key]
            boundingBoxes = []
            labels = []
            if 'annotations' in nextMeta:
                for anno in nextMeta['annotations']:
                    if self.targetFormat == 'xyxy':
                        coords = (
                            anno['x'] - anno['width']/2,
                            anno['y'] - anno['height']/2,
                            anno['x'] + anno['width']/2,
                            anno['y'] + anno['height']/2
                        )
                    else:
                        coords = (
                            anno['x'],
                            anno['y'],
                            anno['width'],
                            anno['height']
                        )
                    label = anno['label']
                    if 'unsure' in anno and anno['unsure'] and self.ignoreUnsure:
                        label = -1      # will automatically be ignored (TODO: also true for models other than RetinaNet?)
                    elif label is None:
                        # this usually does not happen for bounding boxes, but we account for it nonetheless
                        continue
                    else:
                        label = self.labelClassMap[label]
                    
                    # sanity check
                    if coords[2] <= 0 or coords[3] <= 0:
                        continue

                    boundingBoxes.append(coords)
                    labels.append(label)

            # feature vector
            #TODO
            fVec = None
            # if 'fVec' in nextMeta:
            #     fVec = torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32))     #TODO: convert from bytes (torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32)))
            # else:
            #     fVec = None
            
            imagePath = nextMeta['filename']
            self.data.append((boundingBoxes, labels, key, fVec, imagePath))


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        boundingBoxes, labels, imageID, fVec, imagePath = self.data[idx]

        # load image
        img = Image.open(BytesIO(self.fileServer.getFile(imagePath)))

        # convert data
        sz = img.size
        boundingBoxes = torch.tensor(boundingBoxes).clone()
        if len(boundingBoxes):
            if boundingBoxes.dim() == 1:
                boundingBoxes = boundingBoxes.unsqueeze(0)
            boundingBoxes[:,0] *= sz[0]
            boundingBoxes[:,1] *= sz[1]
            boundingBoxes[:,2] *= sz[0]
            boundingBoxes[:,3] *= sz[1]
                
        labels = torch.tensor(labels).long()

        if self.transform is not None and img is not None:
            img, boundingBoxes, labels = self.transform(img, boundingBoxes, labels)

        return img, boundingBoxes, labels, fVec, imageID