'''
    Collation class for segmentation mask models.
    In addition to data organizing, this also handles corrupt/incomplete
    images (i.e., image entries that contain "None"). Those cases are
    filtered and the "intact" flag for the respective image is set to
    False in the database.

    2020 Benjamin Kellenberger
'''

import numpy as np
from PIL import Image
import torch
from util import helpers


class Collator:

    def __init__(self, project, dbConnector):
        self.project = project
        self.dbConnector = dbConnector


    def collate(self, batch):
        imgs = []
        segMasks = []
        imageSizes = []
        imageIDs = []

        for i in range(len(batch)):
            if batch[i][0] is None:
                # corrupt image
                helpers.setImageCorrupt(self.dbConnector, self.project, batch[i][3], True)

            else:
                imgs.append(batch[i][0])
                sz = torch.tensor(batch[i][0].size())
                if batch[i][1] is None:
                    # no segmentation mask; append dummy tensor for "all ignore"
                    segMasks.append(255*torch.ones(size=(sz[1],sz[2],)))           #TODO: offset for "no label" = 255
                else:
                    # segmentation mask present; extract only one band
                    segMask = batch[i][1]
                    if isinstance(segMask, Image.Image):
                        segMask = torch.from_numpy(np.array(segMask)[0,:,:])
                    elif isinstance(segMask, np.ndarray):
                        segMask = torch.from_numpy(segMask[0,:,:])
                    elif isinstance(segMask, torch.Tensor):
                        segMask = segMask[0,...]
                    segMasks.append(segMask.long())
                imageSizes.append(batch[i][2])
                imageIDs.append(batch[i][3])
        
        return torch.stack(imgs), torch.stack(segMasks), imageSizes, imageIDs