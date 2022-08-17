'''
    Detectron2-compliant wrapper for DeepForest models:
    https://github.com/weecology/DeepForest

    2022 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from detectron2.modeling import build_backbone, BACKBONE_REGISTRY, META_ARCH_REGISTRY, Backbone, ShapeSpec
from detectron2.structures import Instances, Boxes
from deepforest import utilities
from deepforest.model import load_backbone, create_anchor_generator, create_model


# @BACKBONE_REGISTRY.register()
# class DeepForestBackbone(Backbone):

#     def __init__(self, cfg, input_shape={}):
#         super(DeepForestBackbone, self).__init__()
#         self.backbone = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
#         self.out_channels = self.backbone.backbone.out_channels
    
#     def forward(self, x):
#         return {'out': self.backbone(x)}
    
#     def output_shape(self):
#         # not needed since our DeepForest (below) is hard-coded, but here for
#         # completeness
#         return {
#             'out': ShapeSpec(channels=2048, stride=1)   #TODO
#         }



@META_ARCH_REGISTRY.register()
class DeepForest(nn.Module):

    def __init__(self, cfg):
        super(DeepForest, self).__init__()

        # load pre-trained DeepForest model if available
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES

        self.release_state_dict = None
        pretrainedName = cfg.MODEL.DEEPFOREST_PRETRAINED
        if pretrainedName == 'deepforest':
            _, self.release_state_dict = utilities.use_release(check_release=True)
            num_classes = 1
            self.names = ('tree',)
        elif pretrainedName == 'birddetector':
            _, self.release_state_dict = utilities.use_bird_release(check_release=True)
            num_classes = 1
            self.names = ('bird',)
        
        self.model = create_model(
            num_classes=num_classes,
            nms_thresh=cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            score_thresh=cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
        )

        if self.release_state_dict is not None:
            self.model.load_state_dict(
                torch.load(self.release_state_dict, map_location='cpu'), strict=False)

        self.out_channels = self.model.backbone.out_channels

    @property
    def device(self):
        return self.model.head.classification_head.cls_logits.weight.device
    
    @property
    def dtype(self):
        return self.model.head.classification_head.cls_logits.weight.dtype

    def forward(self, inputs):
        images = [i['image'].float().to(self.device)/255 for i in inputs]
        targets = None
        if self.training:
            targets = []
            for i in inputs:
                targets.append({
                    'boxes': i['instances'].gt_boxes.tensor.to(self.device),
                    'labels': i['instances'].gt_classes.long().to(self.device)
                })
        out = self.model(images, targets)
        if not self.training:
            if len(out[0]['labels']):
                instances = Instances(image_size=(images[0].size(1), images[0].size(2)))
                instances.pred_classes = out[0]['labels']
                instances.pred_boxes = Boxes(out[0]['boxes'][:,:4])
                instances.scores = out[0]['scores']
                return [{'instances': instances}]
            else:
                return [{}]
        else:
            return out