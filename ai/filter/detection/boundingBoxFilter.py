import numpy as np
from ai.filter import AbstractFilter
from util.helpers import check_args


class BoundingBoxFilter(AbstractFilter):

    def __init__(self, config, dbConnector, fileServer, options):
        super(BoundingBoxFilter, self).__init__(config, dbConnector, fileServer, options)

        # parse properties
        defaultOptions = {
            'box_rule': 'average',          # how to generate the resulting bounding box from overlapping ones. One of {'average', 'intersection', 'union'}
            'min_iou': 0.75,                # minimum IoU between overlapping bboxes to employ filtering
            'class_agnostic': False,        # if True, only overlapping boxes with the same class will be subject to filtering
            'class_assignment': 'mode',     # how to assign class label of overlapping boxes. One of {'mode', 'random'}
            'keep_unsure': True             # if True, "unsure" bounding boxes will directly be appended without modification to output
        }
        self.options = check_args(options, defaultOptions)


    def __box_ious(self, box_a, boxes_b):
        leftX = np.maximum(box_a[0], boxes_b[:,0])
        rightX = np.minimum(box_a[2], boxes_b[:,2])
        topY = np.maximum(box_a[1], boxes_b[:,1])
        bottomY = np.minimum(box_a[3], boxes_b[:,3])

        width = np.clip(rightX - leftX + 1, 0, 1)
        height = np.clip(bottomY - topY + 1, 0, 1)
        intersection = width * height
        union = (box_a[2] - box_a[0])*(box_a[3] - box_a[1]) + \
                (boxes_b[:,2] - boxes_b[:,0])*(boxes_b[:,3] - boxes_b[:,1]) - \
                intersection
        return intersection/union


    def _get_result_box(self, box_a, boxes_b, label_a, labels_b):
        '''
            Returns the resulting bounding box based on the options (i.e., intersection,
            union, or average), as well as the indices of the bounding boxes in 'boxes_b'
            that were involved in it.
        '''
        # convert to numpy
        box_a = np.array(box_a)
        boxes_b = np.array(boxes_b)

        # calc. intersections and unions
        iou = self.__box_ious(box_a, boxes_b)

        # filter for valid ones
        if self.options['class_agnostic']:
            indices = np.where(iou >= self.options['min_iou'])
        else:
            indices = np.where((iou >= self.options['min_iou']) * np.array([int(l==label_a) for l in labels_b]))

        # calculate resulting box
        boxes_concat = np.concatenate([box_a[np.newaxis,:], boxes_b], 0)
        if self.options['box_rule'] == 'average':
            box_out = np.mean(boxes_concat, 0)
        elif self.options['box_rule'] == 'intersection':
            topLeft = boxes_concat.max(0)[0:2]
            bottomRight = boxes_concat.min(0)[2:]
            box_out = np.concatenate([topLeft, bottomRight])
        elif self.options['box_rule'] == 'union':
            topLeft = boxes_concat.min(0)[0:2]
            bottomRight = boxes_concat.max(0)[2:]
            box_out = np.concatenate([topLeft, bottomRight])
        return box_out, indices


    def filter(self, data, **kwargs):

        # prepare result
        data_out = {}

        # iterate over images
        for key in data.keys():
            if not 'annotations' in data[key] or not len(data[key]['annotations']):
                continue

            # prepare all bounding boxes and labels
            bboxes_in = []
            labels_in = []
            ids_in = []
            for annoKey in data[key]['annotations'].keys():
                anno = data[key]['annotations'][annoKey]
                if self.options['keep_unsure'] and 'unsure' in anno and anno['unsure']:
                    data_out[key]['annotations'].append(anno)
                else:
                    bboxes_in.append([
                        anno['x'] - anno['width']/2,
                        anno['y'] - anno['height']/2,
                        anno['x'] + anno['width']/2,
                        anno['y'] + anno['height']/2
                    ])
                    labels_in.append(anno['label'])
                    ids_in.append(annoKey)
            
            # iterate over bounding boxes
            bboxes_out = []
            while len(bboxes_in):
                if len(bboxes_in) == 1:
                    # only one box left; append directly without modification
                    data_out[key]['annotations'].append(data[key]['annotations'][ids_in[0]])

                else:
                    # get resulting box and indices of other bounding boxes that overlap with the next one
                    resultingBox, idx_other = self._get_result_box(bboxes_in[0], bboxes_in[1:], \
                                                                labels_in[0], labels_in[1:])
                    
                    # remove all involved boxes from pool
                    del bboxes_in[0]
                    del bboxes_in[idx_other]    #TODO...