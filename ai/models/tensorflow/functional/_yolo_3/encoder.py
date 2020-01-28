'''Encode object boxes and labels.'''
import math
import numpy as np


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
    def __init__(self, numClasses=1):
        self.anchors = [[116,90],  [156,198],  [373,326],  [30,61], [62,45],  [59,119], [10,13],  [16,30],  [33,23]] # Hardcoded anchor boxes for now
        self.downscale = 32
        self.numClasses = numClasses

    def encode(self, imgs, boxes, labels):
        

        batch_size, net_w, net_h, _ = imgs.shape
	# get image input size
        base_grid_h, base_grid_w = net_h//self.downscale, net_w//self.downscale


        # initialize the inputs and the outputs
        yolo_1 = np.zeros((batch_size, 1*base_grid_h,  1*base_grid_w, 3, 4+1+self.numClasses)) # desired network output 1
        yolo_2 = np.zeros((batch_size, 2*base_grid_h,  2*base_grid_w, 3, 4+1+self.numClasses)) # desired network output 2
        yolo_3 = np.zeros((batch_size, 4*base_grid_h,  4*base_grid_w, 3, 4+1+self.numClasses)) # desired network output 3
        yolos = [yolo_1, yolo_2, yolo_3]

        instance_count = 0
        true_box_index = 0

        # do the logic to fill in the inputs and the output
        for b in range(batch_size):

            # augment input image and fix object's position and size
            all_objs = boxes[b]
            obj_labl = labels[b]
            if all_objs is None:
                continue
            for obj in range(len(obj_labl)):

                box = all_objs[obj]
                label = int(obj_labl[obj])
                # find the best anchor box for this object
                max_anchor = None
                max_index  = -1
                max_iou    = -1

                shifted_box = np.array((0.0, 0.0, box[2], box[3]))

                for i in range(len(self.anchors)):
                    anchor = np.array((0, 0, self.anchors[i][0], self.anchors[i][1]))
                    iou    = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index  = i
                        max_iou    = iou

                # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index//3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                g_center_x = box[0] / float(net_w) * grid_w # sigma(t_x) + c_x
                g_center_y = box[1] / float(net_h) * grid_h # sigma(t_y) + c_y

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(g_center_x))
                grid_y = int(np.floor(g_center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[b, grid_y, grid_x, max_index%3, 0:4] = box
                yolo[b, grid_y, grid_x, max_index%3, 4  ] = 1.
                if label>-1:
                    # class value is ignored if we're unsure of the object 
                    yolo[b, grid_y, grid_x, max_index%3, 5+label] = 1.



        return imgs, [yolo_1, yolo_2, yolo_3]



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
