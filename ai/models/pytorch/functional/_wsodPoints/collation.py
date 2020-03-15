'''
    Collator for point models.

    2019-20 Benjamin Kellenberger
'''

import torch
from .encoder import DataEncoder
from util import helpers


class Collator():
    def __init__(self, project, dbConnector, target_size, encoder):
        self.project = project
        self.dbConnector = dbConnector
        self.target_size = target_size
        self.encoder = encoder

    def collate_fn(self, batch):
        '''
            Collates a batch by recombining its entries into the following variables:
                imgs (tensor): the image tensors of the batch
                loc_targets (tensor): encoded point locations (if available), sized
                                      [batch size, num classes+1, width, height]
                cls_images (tensor): image-wide labels for weakly-supervised models,
                                     sized [batch size]
                fVecs_targets (list): TODO
                imageIDs_targets (list): list of image ID strings
        '''

        # gather batch components
        imgs = []
        points = []
        labels = []
        image_labels = []
        fVecs = []
        imageIDs = []

        for i in range(len(batch)):
            if batch[i][0] is None:
                # corrupt image
                helpers.setImageCorrupt(self.dbConnector, self.project, batch[i][3], True)

            else:
                imgs.append(batch[i][0])
                points.append(batch[i][1])
                labels.append(batch[i][2])
                image_labels.append(batch[i][3])
                fVecs.append(batch[i][4])
                imageIDs.append(batch[i][5])

        # prepare outputs
        loc_targets = []
        cls_images = torch.tensor(image_labels).long()
        fVecs_targets = []
        imageIDs_targets = []

        # assemble outputs
        num_imgs = len(imgs)
        for i in range(num_imgs):

            # spatially explicit points
            input_size = [imgs[i].size(2), imgs[i].size(1)]
            loc_target = self.encoder.encode(points[i], labels[i], input_size, self.target_size)
            loc_targets.append(loc_target)

            # remainder
            fVecs_targets.append(fVecs[i])
            imageIDs_targets.append(imageIDs[i])
        
        imgs = torch.stack(imgs)
        
        return imgs, torch.stack(loc_targets), cls_images, fVecs_targets, imageIDs_targets