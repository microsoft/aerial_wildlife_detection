"""
    RetinaNet implementation that allows using different ResNet backbones
    and features pre-trained on ImageNet.

    2019 Benjamin Kellenberger 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class FPN(nn.Module):
    in_channels = {
        'resnet18': [512, 256, 128, 64],
        'resnet34': [512, 256, 128, 64],
        'resnet50': [2048, 1024, 512, 256],
        'resnet101': [2048, 1024, 512, 256],
        'resnet152': [2048, 1024, 512, 256]
    }

    def __init__(self, backbone, pretrained, out_planes=256, convertToInstanceNorm=False):
        super(FPN, self).__init__()

        if isinstance(backbone, str):
            backbone = getattr(resnet, backbone)
        self.backbone = backbone
        self.pretrained = pretrained
        self.out_planes = out_planes
        self.convertToInstanceNorm = convertToInstanceNorm
        self.fe = backbone(pretrained)

        if self.convertToInstanceNorm:
            for layer in self.fe.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer = nn.InstanceNorm2d(layer.num_features, affine=False, track_running_stats=False)

        self.conv6 = nn.Conv2d(self.in_channels[self.backbone.__name__][0], out_planes, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1)

        # lateral layers
        self.latlayer1 = self._make_lateral(0)
        self.latlayer2 = self._make_lateral(1)
        self.latlayer3 = self._make_lateral(2)

        # top-down layers
        self.toplayer1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)


    def _make_lateral(self, level):
        return nn.Conv2d(self.in_channels[self.backbone.__name__][level], self.out_planes, kernel_size=1, stride=1, padding=0)



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        c1 = self.fe.conv1(x)
        c1 = self.fe.bn1(c1)
        c1 = self.fe.relu(c1)
        c1 = self.fe.maxpool(c1)
        
        # bottom-up stages
        c2 = self.fe.layer1(c1)
        c3 = self.fe.layer2(c2)
        c4 = self.fe.layer3(c3)
        c5 = self.fe.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # top-down stages
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p4, p5, p6, p7




class RetinaNet(nn.Module):

    def __init__(self, labelclassMap, numAnchors=9, backbone='resnet50', pretrained=True, out_planes=256, convertToInstanceNorm=False):
        super(RetinaNet, self).__init__()

        self.labelclassMap = labelclassMap
        self.numClasses = len(labelclassMap.keys())
        self.numAnchors = numAnchors
        self.backbone = backbone
        self.pretrained = pretrained
        self.out_planes = out_planes
        self.convertToInstanceNorm = convertToInstanceNorm
        self.fpn = FPN(self.backbone, self.pretrained, self.out_planes, self.convertToInstanceNorm)

        self.loc_head = self._make_head(self.numAnchors*4, dropout=0.2)
        self.cls_head = self._make_head(self.numAnchors*self.numClasses, dropout=None)


    def getStateDict(self):
        stateDict = {
            'model_state': self.state_dict(),
            'labelclassMap': self.labelclassMap,
            'numAnchors': self.numAnchors,
            'backbone': self.backbone,
            'pretrained': self.pretrained,
            'out_planes': self.out_planes,
            'convertToInstanceNorm': self.convertToInstanceNorm
        }
        return stateDict


    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        numAnchors = (stateDict['numAnchors'] if 'numAnchors' in stateDict else 9)
        backbone = (stateDict['backbone'] if 'backbone' in stateDict else resnet.resnet50)
        pretrained = (stateDict['pretrained'] if 'pretrained' in stateDict else True)
        out_planes = (stateDict['out_planes'] if 'out_planes' in stateDict else 256)
        convertToInstanceNorm = (stateDict['convertToInstanceNorm'] if 'convertToInstanceNorm' in stateDict else False)
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)

        # return model
        model = RetinaNet(labelclassMap, numAnchors, backbone, pretrained, out_planes, convertToInstanceNorm)
        if state is not None:
            model.load_state_dict(state)
        return model


    @staticmethod
    def averageStateDicts(stateDicts):
        model = RetinaNet.loadFromStateDict(stateDicts[0])
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in range(1,len(stateDicts)):
            nextModel = RetinaNet.loadFromStateDict(stateDicts[s])
            state = dict(nextModel.named_parameters())
            for key in state:
                pars[key] += state[key].detach().cpu()
        finalState = stateDicts[-1]
        for key in pars:
            finalState['model_state'][key] = pars[key] / (len(stateDicts))

        return finalState


    @staticmethod
    def averageEpochs(statePaths):
        if isinstance(statePaths, str):
            statePaths = [statePaths]
        model = RetinaNet.loadFromStateDict(torch.load(statePaths[0], map_location=lambda storage, loc: storage))
        if len(statePaths) == 1:
            return model
        
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in statePaths[1:]:
            model = RetinaNet.loadFromStateDict(torch.load(s, map_location=lambda storage, loc: storage))
            state = dict(model.named_parameters())
            for key in state:
                pars[key] += state[key]
        
        finalState = torch.load(statePaths[-1], map_location=lambda storage, loc: storage)
        for key in pars:
            finalState['model_state'][key] = pars[key] / (len(statePaths))
        
        model = RetinaNet.loadFromStateDict(finalState)
        return model

    
    def getParameters(self,freezeFE=False):
        headerParams = list(self.loc_head.parameters()) + list(self.cls_head.parameters())
        if freezeFE:
            return headerParams
        else:
            return list(self.fpn.parameters()) + headerParams

    
    def forward(self, x, isFeatureVector=False):
        if isFeatureVector:
            fms = x
        else:
            fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.numClasses)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)


    def _make_head(self, out_planes, dropout=None):
        layers = []
        for _ in range(4):
            if dropout is not None:
                layers.append(nn.Dropout2d(p=dropout, inplace=False))
            layers.append(nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(self.out_planes, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()