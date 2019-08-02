'''
    Collation function for classification models.

    2019 Benjamin Kellenberger
'''

import torch


def collate(batch):

    imgs = torch.stack([x[0] for x in batch])

    labels = []
    fVecs = []
    imageIDs = []

    for i in range(len(batch)):
        labels.append(batch[i][1] if batch[i][1] is not None else -1)   # account for empty label
        fVecs.append(batch[i][2])
        imageIDs.append(batch[i][3])
    
    return imgs, torch.tensor(labels).long(), fVecs, imageIDs