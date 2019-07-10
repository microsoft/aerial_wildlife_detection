'''
    Pad images and encode targets.
    As for images are of different sizes, we need to pad them to the same size.
    Args:
        batch: (list) of images, cls_targets, loc_targets.
    Returns:
        padded images, stacked cls_targets, stacked loc_targets.

    2019 Benjamin Kellenberger
    Adapted from https://github.com/kuangliu/pytorch-retinanet/blob/master/datagen.py
'''

import torch
from .encoder import DataEncoder


class Collator():
    def __init__(self, inputSize, encoder):
        self.inputSize = inputSize
        self.encoder = encoder

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        fVecs = [x[3] for x in batch]
        imageIDs = [x[4] for x in batch]

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