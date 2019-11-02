'''Encode object boxes and labels.'''
import math
import torch
import numpy as np

from .utils import meshgrid, box_iou, box_nms, change_box_order

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 92*92., 128*128., 192*192.]  # [32*32., 64*64., 128*128., 192*192., 256*256.]    # [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h).float() + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, imgs, boxes, labels):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        if not len(labels):
          # no objects in image
          loc_targets = torch.zeros_like(anchor_boxes)
          cls_targets = torch.LongTensor(anchor_boxes.size(0)).zero_()
          return loc_targets, cls_targets

        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)

        # best-matching anchor per target (to make sure every target gets assigned)
        _, max_ids_target = ious.max(0)
        max_ids[max_ids_target] = torch.arange(boxes.size(0))

        boxes = boxes[max_ids]
        labels = labels[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)

        cls_targets = 1 + labels
        cls_targets[max_ious<self.minIoU_pos] = 0
        ignore = (max_ious>self.maxIoU_neg) & (max_ious<self.minIoU_pos)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1

        # make sure every target gets assigned at least one anchor (the optimal one)
        cls_targets[max_ids_target] = 1 + labels[max_ids_target]

        # sanity check: remove NaNs and Infs
        invalid = (torch.isinf(loc_targets) + \
                  torch.isnan(loc_targets)).sum(1).type(torch.bool)
        loc_targets[invalid,:] = 0
        cls_targets[invalid] = -1

        return loc_targets, cls_targets


    def decode(self, predictions, cls_thresh=0.5, nms_thresh=0.5, return_conf=False):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        batch_size = predictions[0].shape[0]
        num_classes = predictions[0][0][...,5:].shape[-1]

        boxes_out = []
        labels_out = []
        logits_out = []
        for b in range(batch_size):
            new_boxes = np.zeros((0,6))
            logits = np.zeros((0,num_classes))
            for i in range(3):
                netout=predictions[i][b]
                grid_h, grid_w = netout.shape[:2]
                xpos = netout[...,0]
                ypos = netout[...,1]
                wpos = netout[...,2]
                hpos = netout[...,3]

                netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
                class_conf = netout[..., 5:]
                objectness = np.max(netout[...,5:],axis=-1)
                pred_class = np.argmax(netout[...,5:],axis=-1)

                # select only objects above threshold
                indexes = (objectness > cls_thresh)
    
                if np.sum(indexes)==0:
                    continue

                corner1 = np.column_stack((xpos[indexes]-wpos[indexes]/2.0, ypos[indexes]-hpos[indexes]/2.0))
                corner2 = np.column_stack((xpos[indexes]+wpos[indexes]/2.0, ypos[indexes]+hpos[indexes]/2.0))


                new_boxes = np.append(new_boxes, np.column_stack((corner1, corner2, objectness[indexes], pred_class[indexes])),axis=0)
                logits = np.append(logits, class_conf[indexes],axis=0)

            # do nms 
            sorted_indices = np.argsort(-new_boxes[:,4])
            boxes=new_boxes.tolist()

            for i in range(len(sorted_indices)):

                index_i = sorted_indices[i]

                if new_boxes[index_i,4] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= nms_thresh:
                        new_boxes[index_j,4] = 0

            logits = logits[new_boxes[:,4]>0]
            new_boxes = new_boxes[new_boxes[:,4]>0]

            boxes_out.append(new_boxes[:,0:4])
            labels_out.append(new_boxes[:,5])
            if return_conf:
                logits_out.append(logits)

        if return_conf:
          return boxes_out, labels_out, logits_out
        else:
          return boxes_out, labels_out
