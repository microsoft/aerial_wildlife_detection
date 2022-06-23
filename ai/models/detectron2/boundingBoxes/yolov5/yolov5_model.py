'''
    Detectron2-compliant wrapper around YOLOv5 model implementation:
    https://github.com/ultralytics/yolov5

    2022 Benjamin Kellenberger
'''

import yaml
import torch
import torch.nn as nn
from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression
from yolov5.utils.loss import ComputeLoss
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes
from yolov5.utils.general import xyxy2xywh


@META_ARCH_REGISTRY.register()
class YOLOv5(nn.Module):

    def __init__(self, cfg):
        super(YOLOv5, self).__init__()

        yoloCfg = yaml.safe_load(cfg.MODEL.YOLO.dump())     # convert Detectron2 cfg to dict
        ch = cfg.MODEL.get('NUM_CHANNELS', 3)   #TODO
        nc = cfg.MODEL.NUM_CLASSES
        yoloCfg.update({
            'nc': nc,
            'ch': ch
        })

        self.tta = cfg.MODEL.get('TEST_TIME_AUGMENT', False)
        self.model = Model(yoloCfg, ch, nc)
        self.model.hyp = cfg.MODEL.YOLO.HYP       # required for ComputeLoss

        self.nms_conf_thres = self.model.hyp.get('nms_conf_thres', 0.25)
        self.nms_iou_thres = self.model.hyp.get('nms_iou_thres', 0.45)
        self.nms_single_cls = self.model.hyp.get('nms_single_cls', False)
        self.nms_max_det = self.model.hyp.get('nms_max_det', 300)

        self.loss = ComputeLoss(self.model)

        # current label class names; required for model-to-project mapping
        self.names = cfg.get('LABELCLASS_NAMES', [])
        if not len(self.names) and hasattr(self.model, 'names'):
            self.names = self.model.names


    def load_weights(self, weights, strict=False):
        return self.model.load_state_dict(weights, strict)

    
    def to(self, device):
        '''
            Overload because ComputeLoss does not inherit from nn.Module
        '''
        super(YOLOv5, self).to(device)
        self.loss = ComputeLoss(self.model)

    @property
    def device(self):
        return self.model.model[0].conv.weight.device
    
    @property
    def dtype(self):
        return self.model.model[0].conv.weight.dtype
    
    def forward(self, image):

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
        
        else:
            result = {}
            pred, train_pred = self.model(imgs, augment=self.tta)
            pred = non_max_suppression(pred, self.nms_conf_thres, self.nms_iou_thres,
                                        labels=None, multi_label=True, agnostic=self.nms_single_cls,
                                        max_det=self.nms_max_det)
            
            if len(pred) and len(pred[0]):
                instances = Instances(image_size=(imgs.size(2), imgs.size(3)))
                instances.pred_classes = pred[0][:,-1]
                instances.pred_boxes = Boxes(pred[0][:,:4])
                instances.scores = pred[0][:,4]

                result['instances'] = instances
            
        return [result]