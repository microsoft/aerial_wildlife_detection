'''
    PyTorch dataset wrapper, optimized for the AL platform.

    2019 Benjamin Kellenberger
'''

from io import BytesIO
import torch
from torch.utils.data import Dataset
from PIL import Image


class BoundingBoxDataset(Dataset):

    def __init__(self, data, fileServer, targetFormat='xywh', transform=None, ignoreUnsure=False):
        super(BoundingBoxDataset, self).__init__()
        self.fileServer = fileServer
        self.targetFormat = targetFormat
        self.transform = transform
        self.ignoreUnsure = ignoreUnsure
        self.__parse_data(data)

    
    def __parse_data(self, data):
        
        # parse label classes first
        self.classdef = {}          # UUID -> index
        self.classdef_inv = {}      # index -> UUID

        idx = 0
        for key in data['labelClasses']:
            if not 'index' in data['labelClasses'][key]:
                # no relation CNN-to-labelclass defined yet; do it here
                index = idx
                idx += 1
            else:
                index = data['labelClasses'][key]['index']
            self.classdef[key] = index
            self.classdef_inv[index] = key
        
        # parse images
        self.data = []
        for key in data['images']:
            nextMeta = data['images'][key]
            boundingBoxes = []
            labels = []
            if 'annotations' in nextMeta:
                for anno in nextMeta['annotations']:
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
                        label = self.classdef[label]

                    boundingBoxes.append(coords)
                    labels.append(label)
            
            imagePath = nextMeta['filename']
            self.data.append((boundingBoxes, labels, key, imagePath))


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        boundingBoxes, labels, imageID, imagePath = self.data[idx]

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
            if self.targetFormat == 'xyxy':
                a = boundingBoxes[:,:2]
                b = boundingBoxes[:,2:]
                boundingBoxes = torch.cat([a-b/2,a+b/2], 1)

        labels = torch.tensor(labels).long()

        if self.transform is not None:
            img, boundingBoxes, labels = self.transform(img, boundingBoxes, labels)

        return img, boundingBoxes, labels, imageID