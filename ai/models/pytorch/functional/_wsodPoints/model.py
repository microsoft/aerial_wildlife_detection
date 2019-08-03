'''
    Implementation of the heatmap-based model in:
        Kellenberger, Benjamin, Diego Marcos, and Devis Tuia. "When a Few Clicks
        Make All the Difference: Improving Weakly-Supervised Wildlife Detection
        in UAV Images." Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition Workshops. 2019.

    2019 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class WSODPointModel(nn.Module):
    in_channels = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    def __init__(self, labelclassMap, backbone='resnet18', pretrained=True, convertToInstanceNorm=True):
        super(WSODPointModel, self).__init__()

        self.labelclassMap = labelclassMap
        self.numClasses = len(labelclassMap.keys())
        self.backbone = backbone
        self.pretrained = pretrained
        self.convertToInstanceNorm = convertToInstanceNorm

        # init components
        feClass = getattr(resnet, self.backbone)
        self.fe = feClass(self.pretrained)
        self.fe.conv1.stride = (1,1)
        self.fe = nn.Sequential(*list(self.fe.children())[:-2])
        if self.convertToInstanceNorm:
            for layer in self.fe.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer = nn.InstanceNorm2d(layer.num_features,
                                    affine=False, track_running_stats=False)

        self.classifier = nn.Sequential(*[
            nn.Conv2d(in_channels=self.in_channels[self.backbone],
                                    out_channels=1024, 
                                    kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(in_channels=1024,
                                    out_channels=self.numClasses, 
                                    kernel_size=1, bias=True),
            nn.Sigmoid()
        ])
    

    def getStateDict(self):
        stateDict = {
            'model_state': self.state_dict(),
            'labelclassMap': self.labelclassMap,
            'backbone': self.backbone,
            'pretrained': self.pretrained,
            'convertToInstanceNorm': self.convertToInstanceNorm
        }
        return stateDict

    
    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        backbone = (stateDict['backbone'] if 'backbone' in stateDict else 'resnet18')
        pretrained = (stateDict['pretrained'] if 'pretrained' in stateDict else True)
        convertToInstanceNorm = (stateDict['convertToInstanceNorm'] if 'convertToInstanceNorm' in stateDict else False)
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)

        # return model
        model = WSODPointModel(labelclassMap, backbone, pretrained, convertToInstanceNorm)
        if state is not None:
            model.load_state_dict(state)
        return model

    
    @staticmethod
    def averageStateDicts(stateDicts):
        model = WSODPointModel.loadFromStateDict(stateDicts[0])
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in range(1,len(stateDicts)):
            nextModel = WSODPointModel.loadFromStateDict(stateDicts[s])
            state = dict(nextModel.named_parameters())
            for key in state:
                pars[key] += state[key].detach().cpu()
        finalState = stateDicts[-1]
        for key in pars:
            finalState['model_state'][key] = pars[key] / (len(stateDicts))

        return finalState

    
    def getParameters(self,freezeFE=False):
        headerParams = list(self.classifier.parameters())
        if freezeFE:
            return headerParams
        else:
            return list(self.fe.parameters()) + headerParams


    def getOutputSize(self, inputSize):
        if not isinstance(inputSize, torch.Tensor):
            inputSize = torch.tensor(inputSize)
        outputSize = inputSize.clone().float()

        for _ in range(4):
            outputSize = torch.ceil(outputSize / 2.0)

        return outputSize.int()


    def forward(self, x, isFeatureVector=False):
        if isFeatureVector:
            return self.classifier(x)
        else:
            fms = self.fe(x)
            return self.classifier(fms)