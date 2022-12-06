'''
    PyTorch dataset wrapper, optimized for the AL platform.

    2019-20 Benjamin Kellenberger
'''

from io import BytesIO
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class PointsDataset(Dataset):
    '''
        PyTorch-conform wrapper for a dataset containing points and/or image-wide labels (for weak supervision).
        Inputs:
        - data:     A dict with the following entries:
                    - labelClasses: dict with { <labelclass UUID> : { 'index' (optional: labelClass index for this CNN) }}
                                    If no index is provided for each class, it will be generated in key order.
                    - images: dict with
                              { <image UUID> : { 'annotations' : { <annotation UUID> : { 'x', 'y', 'label' (label UUID) }}}}
                    - 'fVec': optional, contains feature vector bytes for image
        - fileServer: Instance that implements a 'getFile' function to load images
        - labelclassMap: a dictionary/LUT with mapping: key = label class UUID, value = index (number) according
                         to the model.
        - transform: Instance of classes defined in 'ai.models.pytorch.functional.transforms.points'. May be None for no transformation at all.
        - ignoreUnsure: if True, all annotations with flag 'unsure' will get a label of -1 (i.e., 'ignore')

        The '__getitem__' function returns the data entry at given index as a tuple with the following contents:
        - img: the loaded and transformed (if specified) image.
        - points: transformed points for the image.
        - labels: labels for each point according to the dataset's 'labelclassMap' LUT (i.e., the labelClass indices).
                  May also be -1; in this case the point is flagged as "unsure."
        - image_label: single long for the image-wide label (for weakly-supervised detection).
                       May also be -1; in this case the image is flagged as "unsure."
        - fVec: a torch tensor of feature vectors (if available; else None)
        - imageID: str, filename of the image loaded
    '''
    def __init__(self, data, fileServer, labelclassMap, transform=None, ignoreUnsure=False):
        super(PointsDataset, self).__init__()
        self.fileServer = fileServer
        self.labelclassMap = labelclassMap
        self.transform = transform
        self.ignoreUnsure = ignoreUnsure
        self.__parse_data(data)

    
    def __parse_data(self, data):
        
        # create inverse label class map as well
        self.labelclassMap_inv = {}
        for key in self.labelclassMap.keys():
            val = self.labelclassMap[key]
            self.labelclassMap_inv[val] = key
        
        # parse images
        self.data = []
        hasUnknownClasses = False
        for key in data['images']:
            nextMeta = data['images'][key]
            points = []
            labels = []
            label_img = -1  # default: ignore
            if 'annotations' in nextMeta:
                # check whether we have an image-wide label or actual points
                if len(nextMeta['annotations']) == 1 and not 'x' in nextMeta['annotations'][0]:
                    # single, non-spatial label
                    anno = nextMeta['annotations'][0]
                    label = anno['label']
                    unsure = (anno['unsure'] if 'unsure' in anno else False)
                    if unsure and self.ignoreUnsure:
                        label_img = -1      # will automatically be ignored
                    elif label is None:
                        # empty label; assign background
                        label_img = 0
                    else:
                        if label not in self.labelclassMap:
                            # unknown class
                            hasUnknownClasses = True
                            continue
                        label_img = self.labelclassMap[label]

                else:
                    # we might have points; iterate over annotations
                    for anno in nextMeta['annotations']:
                        coords = (
                            anno['x'],
                            anno['y'],
                        )
                        label = anno['label']
                        unsure = (anno['unsure'] if 'unsure' in anno else False)
                        if unsure and self.ignoreUnsure:
                            label = -1      # will automatically be ignored
                        elif label is None:
                            # this usually does not happen for points, but we account for it nonetheless
                            continue
                        else:
                            label = self.labelclassMap[label]

                        points.append(coords)
                        labels.append(label)

            # feature vector
            #TODO
            fVec = None
            # if 'fVec' in nextMeta:
            #     fVec = torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32))     #TODO: convert from bytes (torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32)))
            # else:
            #     fVec = None
            
            imagePath = nextMeta['filename']
            self.data.append((points, labels, label_img, key, fVec, imagePath))
        
        if hasUnknownClasses:
            print('WARNING: encountered unknown label classes.')

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        points, labels, label_img, imageID, fVec, imagePath = self.data[idx]

        # load image
        try:
            img = Image.open(BytesIO(self.fileServer.getFile(imagePath))).convert('RGB')
        except Exception:
            print('WARNING: Image {} is corrupt and could not be loaded.'.format(imagePath))
            img = None

        if img is not None:
            points = torch.tensor(points).clone()
            if len(points):
                if points.dim() == 1:
                    points = points.unsqueeze(0)
                points *= torch.tensor(img.size, dtype=points.dtype)
            labels = torch.tensor(labels).long()

            if self.transform is not None:
                img, points, labels = self.transform(img, points, labels)

        return img, points, labels, label_img, fVec, imageID