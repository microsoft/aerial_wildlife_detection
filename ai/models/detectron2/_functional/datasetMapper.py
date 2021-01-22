'''
    Custom, Detectron2-compliant dataset mapper that loads images
    and segmentation masks (if available) from AIDE's FileServer.

    Also translates bounding boxes from relative to absolute coordinates
    if present (relative coordinates are currently not supported in
    Detectron2).

    2020-21 Benjamin Kellenberger
'''

import copy
from io import BytesIO
import base64
import numpy as np
import cv2
import torch
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.structures import BoxMode

class Detectron2DatasetMapper(DatasetMapper):

    def __init__(self, project, fileServer, augmentations, is_train, instance_mask_format='bitmask', recompute_boxes=True):
        super(DatasetMapper, self).__init__()
        self.project = project
        self.fileServer = fileServer
        self.augmentations = augmentations
        if not isinstance(self.augmentations, T.AugmentationList):
            self.augmentations = T.AugmentationList(self.augmentations)
        self.is_train = is_train
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.keypoint_hflip_indices = None  #TODO


    def __call__(self, dataset_dict):
        '''
            Adapted from https://detectron2.readthedocs.io/_modules/detectron2/data/dataset_mapper.html#DatasetMapper
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = cv2.imdecode(np.frombuffer(self.fileServer.getFile(dataset_dict["file_name"]), np.uint8), -1)
        except:
            #TODO: cannot handle corrupt data input here; needs to be done earlier
            print('WARNING: Image {} is corrupt and could not be loaded.'.format(dataset_dict["file_name"]))
            image = None
        # ORIGINAL: image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        utils.check_image_size(dataset_dict, image)

        # convert annotations from relative to XYXY absolute format if needed
        image_shape = image.shape[:2]

        if 'annotations' in dataset_dict:
            for anno in dataset_dict['annotations']:
                if 'bbox_mode' in anno and anno['bbox_mode'] in [
                    BoxMode.XYWH_REL, BoxMode.XYXY_REL
                ]:
                    if anno['bbox_mode'] == BoxMode.XYWH_REL:
                        anno['bbox'][0] -= anno['bbox'][2]/2
                        anno['bbox'][1] -= anno['bbox'][3]/2
                        anno['bbox'][2] += anno['bbox'][0]
                        anno['bbox'][3] += anno['bbox'][1]
                    anno['bbox'][0] *= image_shape[0]   #TODO: check order
                    anno['bbox'][1] *= image_shape[1]   #TODO: check order
                    anno['bbox'][2] *= image_shape[0]   #TODO: check order
                    anno['bbox'][3] *= image_shape[1]   #TODO: check order
                    anno['bbox_mode'] = BoxMode.XYXY_ABS

        if "segmentationMask" in dataset_dict:
            try:
                raster = np.frombuffer(base64.b64decode(dataset_dict['segmentationMask']), dtype=np.uint8)
                sem_seg_gt = np.reshape(raster, image_shape)    #TODO: check format
            except:
                print('WARNING: Segmentation mask for image "{}" could not be loaded or decoded.'.format(dataset_dict["file_name"]))
                sem_seg_gt = None
            # ORIGINAL: sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            # for anno in dataset_dict["annotations"]:
                # if not self.use_instance_mask:
                #     anno.pop("segmentation", None)
                # if not self.use_keypoint:
                #     anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes and len(instances) and hasattr(instances, 'gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances) #TODO: do we want that? Maybe limit to width and height assignment to dict entry...
        return dataset_dict