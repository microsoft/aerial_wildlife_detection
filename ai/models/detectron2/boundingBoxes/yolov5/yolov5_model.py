'''
    Detectron2-compliant wrapper around YOLOv5 model implementation:
    https://github.com/ultralytics/yolov5

    2022-23 Benjamin Kellenberger
'''

import yaml
import torch
from torch import nn
from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.general import xyxy2xywh
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes



@META_ARCH_REGISTRY.register()
class YOLOv5(nn.Module):
    '''
        Detectron2-compliant wrapper for YOLOv5 models.
    '''

    def __init__(self, cfg):
        super().__init__()

        yolo_cfg = yaml.safe_load(cfg.MODEL.YOLO.dump())        # convert Detectron2 cfg to dict
        num_channels = cfg.MODEL.get('NUM_CHANNELS', 3)         #TODO
        num_classes = cfg.MODEL.NUM_CLASSES
        yolo_cfg.update({
            'nc': num_classes,
            'ch': num_channels
        })

        self.tta = cfg.MODEL.get('TEST_TIME_AUGMENT', False)
        self.model = Model(yolo_cfg, num_channels, num_classes)
        self.model.hyp = cfg.MODEL.YOLO.HYP       # required for ComputeLoss

        self.nms_conf_thres = self.model.hyp.get('nms_conf_thres', 0.25)
        self.nms_iou_thres = self.model.hyp.get('nms_iou_thres', 0.45)
        self.nms_single_cls = self.model.hyp.get('nms_single_cls', False)
        self.nms_max_det = self.model.hyp.get('nms_max_det', 300)
        self.autobalance = self.model.hyp.get('loss_autobalance', False)

        self.loss = ComputeLoss(self.model, autobalance=self.autobalance)

        # current label class names; required for model-to-project mapping
        self.names = cfg.get('LABELCLASS_NAMES', [])
        if len(self.names) == 0 and hasattr(self.model, 'names'):
            self.names = self.model.names


    def load_weights(self, weights, strict=False):
        return self.model.load_state_dict(weights, strict)


    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        try:
            status = super().load_state_dict(state_dict, strict=False)
            if len(status.missing_keys) > 0 or len(status.unexpected_keys) > 0:
                # try loading within model
                status_model = self.model.load_state_dict(state_dict, strict=False)

                if len(status.missing_keys < status_model.missing_keys):
                    return super().load_state_dict(state_dict, strict=False)
                return status_model
            return status
        except Exception:
            return self.model.load_state_dict(state_dict, strict=False)


    def to(self, device):
        '''
            Overload because ComputeLoss does not inherit from nn.Module.
        '''
        super().to(device)
        self.loss = ComputeLoss(self.model)


    @property
    def device(self):
        return self.model.model[0].conv.weight.device


    @property
    def dtype(self):
        '''
            Convenience/compatibility function to return typical nmeric data type.
        '''
        return self.model.model[0].conv.weight.dtype


    def forward(self, image: torch.Tensor) -> list:
        '''
            Forward pass of the YOLOv5 model.
        '''
        imgs = torch.stack([i['image'] for i in image]).type(self.dtype).to(self.device) / 255

        if self.training:
            # assemble targets
            targets = []
            for idx, img in enumerate(image):
                inst = img['instances']
                target = torch.cat((
                    idx*torch.ones_like(inst.gt_classes).unsqueeze(1),
                    inst.gt_classes.unsqueeze(1),
                    xyxy2xywh(inst.gt_boxes.tensor)), 1)
                targets.append(target)
            targets = torch.cat(targets, 0).to(self.device)

            train_pred = self.model(imgs, augment=self.tta)
            loss = self.loss([x.float() for x in train_pred], targets)[0]   # box, obj, cls
            return {'loss': loss}

        # else prediction
        result = {}
        pred, train_pred = self.model(imgs, augment=self.tta)
        pred = non_max_suppression(pred, self.nms_conf_thres, self.nms_iou_thres,
                                    labels=None, multi_label=True, agnostic=self.nms_single_cls,
                                    max_det=self.nms_max_det)

        if len(pred) > 0 and len(pred[0]) > 0:
            instances = Instances(image_size=(imgs.size(2), imgs.size(3)))
            instances.pred_classes = pred[0][:,-1]
            instances.pred_boxes = Boxes(pred[0][:,:4])
            instances.scores = pred[0][:,4]

            result['instances'] = instances

        return [result]
