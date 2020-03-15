'''
    Collation class for classification models.
    In addition to data organizing, this also handles corrupt/incomplete
    images (i.e., image entries that contain "None"). Those cases are
    filtered and the "intact" flag for the respective image is set to
    False in the database.

    2019-20 Benjamin Kellenberger
'''

import torch
from util import helpers


class Collator:

    def __init__(self, project, dbConnector):
        self.project = project
        self.dbConnector = dbConnector


    def collate(self, batch):
        imgs = []
        labels = []
        fVecs = []
        imageIDs = []

        for i in range(len(batch)):
            if batch[i][0] is None:
                # corrupt image
                helpers.setImageCorrupt(self.dbConnector, self.project, batch[i][3], True)

            else:
                imgs.append(batch[i][0])
                labels.append(batch[i][1] if batch[i][1] is not None else -1)   # account for empty label
                fVecs.append(batch[i][2])
                imageIDs.append(batch[i][3])
        
        return torch.stack(imgs), torch.tensor(labels).long(), fVecs, imageIDs