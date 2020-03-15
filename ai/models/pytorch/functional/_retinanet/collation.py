'''
    Pad images and encode targets.
    As for images are of different sizes, we need to pad them to the same size.
    Args:
        batch: (list) of images, cls_targets, loc_targets.
    Returns:
        padded images, stacked cls_targets, stacked loc_targets.

    2019-20 Benjamin Kellenberger
    Adapted from https://github.com/kuangliu/pytorch-retinanet/blob/master/datagen.py
'''

import torch
from .encoder import DataEncoder
from util import helpers


class Collator():
    def __init__(self, project, dbConnector, inputSize, encoder):
        self.project = project
        self.dbConnector = dbConnector
        self.inputSize = inputSize
        self.encoder = encoder

    def collate_fn(self, batch):
        imgs = []
        boxes = []
        labels = []
        fVecs = []
        imageIDs = []
        for idx in range(len(batch)):
            if batch[idx][0] is None:
                # corrupt image
                helpers.setImageCorrupt(self.dbConnector, self.project, batch[idx][4], True)
            
            else:
                imgs.append(batch[idx][0])
                boxes.append(batch[idx][1])
                labels.append(batch[idx][2])
                fVecs.append(batch[idx][3])
                imageIDs.append(batch[idx][4])

        h, w = self.inputSize
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        fVecs_targets = []
        imageIDs_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            fVecs_targets.append(fVecs[i])
            imageIDs_targets.append(imageIDs[i])

        
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), fVecs_targets, imageIDs_targets