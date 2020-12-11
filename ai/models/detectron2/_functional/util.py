'''
    Utility functions around Detectron2 wrappers.

    2020 Benjamin Kellenberger
'''
import torch

def intersectionOverUnion(boxes_a, boxes_b):
    '''
        Calculates the IoU (Jaccard index, etc.) for two sets of
        Detectron2 instances. Returns an NxM matrix of scores for
        N instances in "boxes_a" and M instances in "boxes_b".
        Boxes are supposed to be in XYXY (left top right bottom)
        format.
    '''
    boxes_b = boxes_b.to(boxes_a.device)

    lt = torch.max(boxes_a[:,None,:2], boxes_b[:,:2])  # [N,M,2]
    rb = torch.min(boxes_a[:,None,2:], boxes_b[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (boxes_a[:,2]-boxes_a[:,0]+1) * (boxes_a[:,3]-boxes_a[:,1]+1)  # [N,]
    area2 = (boxes_b[:,2]-boxes_b[:,0]+1) * (boxes_b[:,3]-boxes_b[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)

    return iou