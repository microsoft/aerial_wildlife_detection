'''
    PyTorch dataset wrapper for image classification datasets.

    2019 Benjamin Kellenberger
'''

from io import BytesIO
import torch
from torch.utils.data import Dataset
from PIL import Image


class LabelsDataset(Dataset):

    def __init__(self, data, fileServer, labelclassMap, transform, ignoreUnsure=False, **kwargs):
        super(LabelsDataset, self).__init__()
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
            label_img = 0       #TODO: default: background
            if 'annotations' in nextMeta:
                # only extract first annotation, since we're dealing with image classification
                if len(nextMeta['annotations']):
                    anno = nextMeta['annotations'][0]
                    label = anno['label']
                    unsure = (anno['unsure'] if 'unsure' in anno else False)
                    if label is None or (self.ignoreUnsure and unsure):
                        continue
                    else:
                        if label not in self.labelclassMap:
                            # unknown class
                            hasUnknownClasses = True
                            continue
                        label_img = self.labelclassMap[label]

            # feature vector
            #TODO
            fVec = None
            # if 'fVec' in nextMeta:
            #     fVec = torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32))     #TODO: convert from bytes (torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32)))
            # else:
            #     fVec = None
            
            imagePath = nextMeta['filename']
            self.data.append((label_img, key, fVec, imagePath))
        
        if hasUnknownClasses:
            print('WARNING: encountered unknown label classes.')

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):

        label, imageID, fVec, imagePath = self.data[idx]

        # load image
        img = Image.open(BytesIO(self.fileServer.getFile(imagePath))).convert('RGB')

        if self.transform is not None and img is not None:
            img = self.transform(img)

        return img, label, fVec, imageID