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
        numRest = len(batch[0]) - 3
        if numRest:
            rest = list(map(list, zip(*[x[3:] for x in batch])))

        h, w = self.inputSize
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        rest_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            if numRest:
                rest_targets.append(rest[i])

        if numRest:
            returns = [inputs, torch.stack(loc_targets), torch.stack(cls_targets)]
            returns.extend(rest)
            return tuple(returns)
            # return inputs, torch.stack(loc_targets), torch.stack(cls_targets), rest
        else:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)