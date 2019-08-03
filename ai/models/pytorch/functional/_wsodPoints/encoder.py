'''
    Encodes point coordinates into a target label grid.

    2019 Benjamin Kellenberger
'''

import torch

class DataEncoder:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    
    def encode(self, points, labels, input_size, target_size):
        '''
            Translates points to a grid defined by the model's prediction size.
            Shifts all label indices by 1, so that label zero corresponds to the
            "ignore" class.

            Args:
                points (tensor): coordinates of the points; sized [Nx2]
                labels (tensor): labels (long) of the points; sized [N]
                input_size (int/tuple/tensor): original image dimensions (width, height)
                target_size (int/tuple/tensor): model prediction grid size (width, height)
            
            Returns:
                loc_targets (tensor): encoded target grid of size [CxWxH], with C =
                                      #classes + 1, W, H = width, height (target_size)
        '''

        if isinstance(input_size, int):
            input_size = torch.tensor([input_size, input_size], dtype=torch.int)
        elif not isinstance(input_size, torch.Tensor):
            input_size = torch.tensor(input_size, dtype=torch.int)
        else:
            input_size = input_size.int()

        if isinstance(target_size, int):
            target_size = torch.tensor([target_size, target_size], dtype=torch.int)
        elif not isinstance(target_size, torch.Tensor):
            target_size = torch.tensor(target_size, dtype=torch.int)
        else:
            target_size = target_size.int()
        
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # prepare target grid
        loc_targets = torch.FloatTensor(self.num_classes + 1, target_size[0].item(), target_size[1].item()).zero_()

        if not len(labels):
            return loc_targets

        # distribute labels in grid
        positions = (points.float() / input_size.float() * target_size.float()).long()
        #TODO: remove invalid positions
        
        # label classes
        positions = torch.cat((labels.long()+1, positions), 1)

        # assign
        loc_targets[positions[:,0], positions[:,1], positions[:,2]] = 1

        return loc_targets

    
    def decode(self, loc_preds, min_conf=0.5, nms_dist=2):
        '''
            Extracts coordinates from a prediction grid.

            Args:
                loc_preds (tensor): prediction coordinate; sized [CxWxH], with C =
                                    #classes, W, H = width, height
                min_conf (float): minimum value entries in loc_preds must have to
                                  be considered as potential predictions.
            
            Returns:
                points (tensor): extracted points whose predicted value is >= min_conf,
                                 converted back to relative values (convention: the cen-
                                 ter of the grid cell the point falls into denotes its
                                 position)
                labels (tensor): arg max of the predicted values (i.e., class label)
                confidences (tensor): predicted values; sized [NxC], with N = number of
                                      predicted points, C = number of classes
        '''

        if loc_preds.dim() == 2:
            loc_preds = loc_preds.unsqueeze(0)
        loc_preds = loc_preds.float()

        # locate valid points
        loc_preds_flat = loc_preds.view(self.num_classes, -1)
        positions_flat = torch.nonzero(loc_preds_flat >= min_conf)
        confidences = torch.index_select(loc_preds_flat, 1, positions_flat[:,1]).permute(1,0)
        points = (torch.nonzero(loc_preds >= min_conf)[:,1:]).cpu()
        labels = (torch.argmax(confidences, 1).squeeze()).cpu()

        # perform NMS
        #TODO

        # convert points to relative format
        sz = torch.tensor(loc_preds.size(), dtype=torch.float)
        points = (points.float() + 0.5) / sz[1:]

        return points, labels, confidences