'''
    Point model loss function that supports both spatially explicit targets (i.e., points)
    and image-wide labels for weakly-supervised object detection.

    For the weak supervision, see:
        Kellenberger, Benjamin, Diego Marcos, and Devis Tuia. "When a Few Clicks Make All the 
            Difference: Improving Weakly-Supervised Wildlife Detection in UAV Images." Procee-
            dings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops.
            2019.

    2019 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointsLoss(nn.Module):

    def __init__(self, background_weight=1.0):
        super(PointsLoss, self).__init__()
        self.background_weight = background_weight

    
    def _wsod_loss(self, x, y):
        '''
            Weakly-supervised loss for point models. Applies a loss as follows:
                l(x, y) =
                    0               if y_c = 1 and sum(x_c) >= 1
                    l_Huber(x, y)   otherwise
                
                with:
                    l_Huber(x, y) = mean(smooth_l1_loss(sum(x_c, y_c))) for all c
            
            Args:
                x (tensor): predicted heatmap of size [batch_size, num_classes, ..., ...]
                y (tensor): target class labels of size [batch_size], taking values in
                            range [-1, num_classes + 1].
                            The loss for cases where y = -1 will be set to zero.
                            A value of 0 denotes the absence of all classes, which means
                            that class indices are shifted by one.
        '''

        sz = x.size()

        # get heatmap sum
        x_sum = torch.sum(x.view(sz[0], sz[1], -1), -1)     # [batch_size, num_classes]

        # construct target
        valid = (y > 0)     # exclude unsure and zero classes for target
        y_target = torch.FloatTensor(sz[0], sz[1]).zero_()
        y_target[valid].scatter_(1, y[valid]-1, 1)

        # calc. base loss
        loss = F.smooth_l1_loss(x_sum, y_target, reduction='none')

        # mask case A
        case_a = (x_sum >= 1.0) * (y_target >= 1.0)     #TODO: batch size
        loss[case_a] = 0.0

        # mask unsure cases
        unsure = (y == -1)
        loss[unsure] = 0.0      #TODO: ditto

        # normalize and return
        return torch.mean(loss)



    def forward(self, loc_preds, loc_targets, cls_images):
        '''
            Computes the point loss as follows:
                - for every position where cls_images = -1 (ignore), the loss corresponds
                  to SmoothL1Loss(loc_preds, loc_targets)
                - if cls_images != -1, the _wsod_loss(loc_preds, cls_images) is calculated
                  instead (weak supervision).
            
            Args:
                loc_preds (tensor): predicted locations, sized [batch_size, num_classes, w, h]
                loc_targets (tensor): target locations, sized [batch_size, num_classes + 1, w, h]
                cls_images (tensor): image-wide labels, sized [batch_size]
        '''

        # find images where a direct spatial loss can be calculated
        valid = (cls_images > -1)

        # calculate spatial loss
        loss_spatial = F.smooth_l1_loss(loc_preds[valid,...], loc_targets[valid,1:,...], reduction='none')

        # apply background weight
        background = (loc_targets[valid,1:,...] == 0)   #TODO: sum along classes
        loss_spatial[background] *= self.background_weight

        # mask unsure cases
        unsure = loc_targets[valid,0,...] > 0
        loss_spatial[valid,unsure,...] = 0
        loss_spatial = torch.mean(loss_spatial)

        # weak supervision
        valid = ~valid
        loss_wsod = self._wsod_loss(loc_preds[valid,...], cls_images[valid,...])

        return (loss_spatial + loss_wsod) / 2