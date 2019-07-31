'''
    PyTorch dataset wrapper for image classification datasets.

    2019 Benjamin Kellenberger
'''

from io import BytesIO
import torch
from torch.utils.data import Dataset
from PIL import Image


class ClassificationDataset(Dataset):

    def __init__(self, data, fileServer, labelclassMap, transform, ignoreUnsure=False, **kwargs):
        super(ClassificationDataset, self).__init__()
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
        for key in data['images']:
            nextMeta = data['images'][key]
            labels = []
            if 'annotations' in nextMeta:
                for anno in nextMeta['annotations']:
                    label = anno['label']
                    if label is None or ('unsure' in anno and anno['unsure'] and self.ignoreUnsure):
                        continue
                    else:
                        label = self.labelclassMap[label]

                    labels.append(label)

            # feature vector
            #TODO
            fVec = None
            # if 'fVec' in nextMeta:
            #     fVec = torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32))     #TODO: convert from bytes (torch.from_numpy(np.frombuffer(anno['fVec'], dtype=np.float32)))
            # else:
            #     fVec = None
            
            imagePath = nextMeta['filename']
            self.data.append((labels, key, fVec, imagePath))
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):

        labels, imageID, fVec, imagePath = self.data[idx]

        # load image
        img = Image.open(BytesIO(self.fileServer.getFile(imagePath)))

        # convert data     
        labels = torch.tensor(labels).long()

        if self.transform is not None and img is not None:
            img = self.transform(img)

        return img, labels, fVec, imageID