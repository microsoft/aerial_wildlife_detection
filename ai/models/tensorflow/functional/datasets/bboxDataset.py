'''
    PyTorch dataset wrapper, optimized for the AL platform.

    2019 Benjamin Kellenberger
'''

from io import BytesIO
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

class BoundingBoxesDataset(Sequence):
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
        - labelclassMap: a dictionary/LUT with mapping: key = label class UUID, value = index (number) according
                         to the model.
        - targetFormat: str for output bounding box format, either 'xywh' (xy = center coordinates, wh = width & height)
                        or 'xyxy' (top left and bottom right coordinates)
        - transform: Instance of classes defined in 'ai.models.pytorch.functional.transforms.boundingBoxes'. May be None for no transformation at all.
        - ignoreUnsure: if True, all annotations with flag 'unsure' will get a label of -1 (i.e., 'ignore')

        The '__getitem__' function returns the data entry at given index as a tuple with the following contents:
        - img: the loaded and transformed (if specified) image.
        - boundingBoxes: transformed bounding boxes for the image.
        - labels: labels for each bounding box according to the dataset's 'labelclassMap' LUT (i.e., the labelClass indices).
                  May also be -1; in this case the bounding box is flagged as "unsure."
        - imageID: str, filename of the image loaded
    '''
    def __init__(self, data, fileServer, labelclassMap, batch_size=4, shuffle=False, targetFormat='xywh', transform=None, ignoreUnsure=False, encoder=None):
        self.fileServer = fileServer
        self.labelclassMap = labelclassMap
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.targetFormat = targetFormat
        self.transform = transform
        self.ignoreUnsure = ignoreUnsure
        self.encoder = encoder
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
                    unsure = (anno['unsure'] if 'unsure' in anno else False)
                    if unsure and self.ignoreUnsure:
                        label = -1      # will automatically be ignored
                    elif label is None:
                        # this usually does not happen for bounding boxes, but we account for it nonetheless
                        continue
                    else:
                        if label not in self.labelclassMap:
                            # unknown class
                            hasUnknownClasses = True
                            continue
                        label = self.labelclassMap[label]
                    
                    # sanity check
                    if coords[2] <= 0 or coords[3] <= 0:
                        continue

                    boundingBoxes.append(coords)
                    labels.append(label)

            
            imagePath = nextMeta['filename']
            self.data.append((boundingBoxes, labels, key, imagePath))
        
        if hasUnknownClasses:
            print('WARNING: encountered unknown label classes.')

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.batch_size))           

    
    def __getitem__(self, idx):

        l_bound = idx*self.batch_size
        r_bound = min(len(self.data), (idx+1)*self.batch_size)

        img_b = []
        boundingBoxes_b = []
        labels_b = []
        imageID_b = []

        for idx_b in range(l_bound, r_bound):
            boundingBoxes, labels, imageID, imagePath = self.data[idx_b]

            # load image
            img = Image.open(BytesIO(self.fileServer.getFile(imagePath))).convert('RGB')

            # convert data
            sz = img.size
            boundingBoxes = np.array(boundingBoxes)
            if len(boundingBoxes):
                if boundingBoxes.ndim == 1:
                    boundingBoxes = boundingBoxes[None,...]
                boundingBoxes[:,0] *= sz[0]
                boundingBoxes[:,1] *= sz[1]
                boundingBoxes[:,2] *= sz[0]
                boundingBoxes[:,3] *= sz[1]
                    
            if self.transform is not None and img is not None:
                img, boundingBoxes, labels = self.transform(img, boundingBoxes, labels)

            img = np.asarray(img,dtype=np.float32)/255.0

            img_b.append(img)
            boundingBoxes_b.append(boundingBoxes)
            labels_b.append(labels)
            imageID_b.append(imageID)

        img_b = np.asarray(img_b)

        if self.encoder is not None:
            # encode for training
            return self.encoder(img_b, boundingBoxes_b, labels_b)
        else:
            return img_b, boundingBoxes_b, labels_b, imageID_b
