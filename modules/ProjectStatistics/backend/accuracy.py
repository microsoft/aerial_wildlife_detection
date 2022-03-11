'''
    Formulas for accuracy calculation.

    2022 Benjamin Kellenberger
'''

import numpy as np



def precision_recall_f1(tp, fp, fn):
    '''
        Calculates precision, recall and F1 score based on true positives, false
        positives, and false negatives.
    '''
    try:
        prec = (tp / float(tp+fp))
    except:
        prec = 0.0
    try:
        rec = (tp / float(tp+fn))
    except:
        rec = 0.0
    try:
        f1 = 2 * prec*rec / (prec+rec)
    except:
        f1 = 0.0
    return prec, rec, f1
    


def intersection_over_union(boxesA, boxesB):
    '''
        Receives two NumPy ndarrays of size 4 or Nx4 and calculates the
        intersection-over-union (IoU; Jaccard index) between all. Each row in
        each array represents one bounding box, with format [x, y, width,
        height], where x and y are center coordinates. Returns an NxM ndarray
        for n rows in "boxesA" and m rows in "boxesB", containing IoU values
        accordingly.
    '''
    if boxesA.ndim == 1:
        boxesA = boxesA[np.newaxis,:]
    if boxesB.ndim == 1:
        boxesB = boxesB[np.newaxis,:]

    # convert to XYXY (top left, bottom right) format
    boxesA[:,0] -= boxesA[:,2]/2.0
    boxesA[:,1] -= boxesA[:,3]/2.0
    boxesA[:,2] += boxesA[:,0]
    boxesA[:,3] += boxesA[:,1]

    boxesB[:,0] -= boxesB[:,2]/2.0
    boxesB[:,1] -= boxesB[:,3]/2.0
    boxesB[:,2] += boxesB[:,0]
    boxesB[:,3] += boxesB[:,1]

    # iterate
    iou = np.zeros((len(boxesA), len(boxesB)))
    for b in range(len(boxesA)):
        tlMin = np.minimum(boxesA[b,:2], boxesB[:,:2])
        tlMax = np.maximum(boxesA[b,:2], boxesB[:,:2])
        brMin = np.minimum(boxesA[b,2:], boxesB[:,2:])
        brMax = np.maximum(boxesA[b,2:], boxesB[:,2:])

        intersection = np.prod(brMin - tlMax, axis=1)
        intersection[np.any(brMin - tlMax <= 0, 1)] = 0
        union = np.prod((brMin - tlMin), axis=1) + np.prod((brMax - tlMax), axis=1) - intersection

        iou[b,:] = intersection / union
    
    return iou



'''
    Formulas per annotation type
'''
def statistics_labels(pred, target):
    '''
        Calculates statistical accuracy measures for image labels, including:
        - overall accuracy
        - TODO: more
    '''
    pass


def statistics_points(pred, target, maxDistance):
    '''
        Calculates statistical accuracy measures for points. "maxDistance"
        denotes the maximum permitted spatial displacement of predictions versus
        targets for which a prediction can still be counted potentially correct.
    '''
    pass


def statistics_boundingBoxes(pred, target, minIoU=0.5):
    '''
        Calculates statistical accuracy measures for bounding boxes. "minIoU"
        denotes the minimum Intersection-over-Union (IoU; Jaccard index)
        required for predictions with the closest ground truth to be considered
        potentially correct.
        TODO: penalize multiple predictions per ground truth target
    '''

    stats = {
        'img': {}
    }

    # global stats
    avgIoU_global = 0
    avgIoU_correct_global = 0
    TP, FP, FN = 0, 0, 0
    numImgs_match = 0           # no. images where there's at least one TP (for avg. IoU correct calc.)

    for imgKey in pred.keys():
        pred_img = pred[imgKey]
        target_img = target[imgKey]

        # extract annotations
        ids_pred = []
        labels_pred, labels_target = [], []
        bboxes_pred, bboxes_target = [], []

        if 'annotations' not in pred_img:
            pred_img['annotations'] = []
        if 'annotations' not in target_img:
            target_img['annotations'] = []

        for anno in pred_img['annotations']:
            ids_pred.append(anno['id'])
            labels_pred.append(str(anno['label']))
            geom = anno.get('geometry', anno)
            bboxes_pred.append([
                geom['x'], geom['y'],
                geom['width'], geom['height']
            ])
        for anno in target_img['annotations']:
            labels_target.append(str(anno['label']))
            geom = anno.get('geometry', anno)
            bboxes_target.append([
                geom['x'], geom['y'],
                geom['width'], geom['height']
            ])
        labels_pred, labels_target = np.array(labels_pred), np.array(labels_target)
        labels_pred = np.repeat(labels_pred[:,np.newaxis], len(labels_target), 1)
        labels_target = np.repeat(labels_target[np.newaxis,:], len(labels_pred), 0)

        bboxes_pred, bboxes_target = np.array(bboxes_pred), np.array(bboxes_target)

        if len(bboxes_pred) and len(bboxes_target):
            iou = intersection_over_union(bboxes_pred, bboxes_target)
            avgIoU_global += np.mean(np.max(iou, 1)).tolist()       # taken across predictions

            # candidates: correct label and sufficient IoU
            valid = (iou>=minIoU) * (labels_pred == labels_target)

            # iterate in order, assigning correct predictions starting from highest IoU
            assignments_pred = np.zeros(valid.shape[0], dtype=np.uint8)          # pred can be 1 (TP) or 2 (FP)
            assignments_target = np.zeros(valid.shape[1], dtype=np.uint8)        # target can be 1 (TP) or 3 (FN)

            avgIoU_correct = 0
            count_correct = 0

            idx_pred, idx_target = np.where(valid)
            order = np.argsort(iou[valid])[::-1]
            for o in order:
                idx_pred_, idx_target_ = idx_pred[o], idx_target[o]
                if assignments_target[idx_target_] == 1:
                    # target already predicted; label as false positive
                    assignments_pred[idx_pred_] = 2
                else:
                    # target not predicted yet
                    if assignments_pred[idx_pred_] == 1:
                        # prediction already accounts for another target; label as false positive
                        assignments_pred[idx_pred_] = 2
                    else:
                        # prediction not associated with any other target; label as true positive
                        assignments_pred[idx_pred_] = 1
                        assignments_target[idx_target_] = 1
                        avgIoU_correct += iou[valid][o].tolist()
                        count_correct += 1
            
            if count_correct > 0:
                avgIoU_correct /= count_correct
            avgIoU_correct_global += avgIoU_correct

            # assign remaining predictions and targets
            assignments_pred[assignments_pred == 0] = 2
            assignments_target[assignments_target == 0] = 3

            # construct lists of TP and FP
            tp, fp = [], []
            for idx in range(len(assignments_pred)):
                if assignments_pred[idx] == 2:
                    fp.append(ids_pred[idx])
                elif assignments_pred[idx] == 1:
                    tp.append(ids_pred[idx])

            num_tp = np.sum(assignments_target==1).tolist()
            if num_tp > 0: numImgs_match += 1
            TP += num_tp
            FP += np.sum(assignments_pred==2).tolist()
            FN += np.sum(assignments_target==3).tolist()
        
        elif not len(bboxes_target):
            # all predictions are FP
            FP += len(bboxes_pred)
            fp = ids_pred
            tp = []
            avgIoU_correct = 0
        elif not len(bboxes_pred):
            # all targets are FN
            FN += len(bboxes_target)
            fp, tp = [], []
            avgIoU_correct = 0

        stats['img'][imgKey] = {
            'tp': tp,
            'fp': fp,
            'avg_iou_correct': avgIoU_correct
        }
    
    # global stats
    prec, rec, f1 = precision_recall_f1(TP, FP, FN)
    stats['global'] = {
        'num_img': len(pred.keys()),
        'num_imgs_match': numImgs_match,
        'num_tp': TP,
        'num_fp': FP,
        'num_fn': FN,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'avg_iou': avgIoU_global / len(pred),
        'avg_iou_correct': avgIoU_correct_global / len(pred)
    }
    return stats