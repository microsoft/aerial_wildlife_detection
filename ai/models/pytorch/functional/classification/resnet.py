'''
    Generic ResNet model for image classification.

    2019 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
from torchvision.models import resnet


class ResNet(nn.Module):
    in_features = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    def __init__(self, labelclassMap, featureExtractor='resnet50', pretrained=True):
        super(ResNet, self).__init__()

        self.labelclassMap = labelclassMap
        self.featureExtractor = featureExtractor
        self.pretrained = pretrained

        # create actual model
        if isinstance(featureExtractor, str):
            featureExtractor = getattr(resnet, featureExtractor)
        self.fe = featureExtractor(pretrained)
        self.fe = nn.Sequential(*list(self.fe.children())[:-1])

        self.classifier = nn.Linear(in_features=self.in_features[featureExtractor.__name__],
                                    out_features=len(labelclassMap.keys()), bias=True)
    

    def getStateDict(self):
        stateDict = {
            'model_state': self.state_dict(),
            'labelclassMap': self.labelclassMap,
            'featureExtractor': self.featureExtractor,
            'pretrained': self.pretrained
        }
        return stateDict


    @staticmethod
    def loadFromStateDict(stateDict):
        # parse args
        labelclassMap = stateDict['labelclassMap']
        featureExtractor = (stateDict['featureExtractor'] if 'featureExtractor' in stateDict else 'resnet50')
        pretrained = (stateDict['pretrained'] if 'pretrained' in stateDict else True)
        state = (stateDict['model_state'] if 'model_state' in stateDict else None)

        # return model
        model = ResNet(labelclassMap, featureExtractor, pretrained)
        if state is not None:
            model.load_state_dict(state)
        return model

    
    @staticmethod
    def averageStateDicts(stateDicts):
        model = ResNet.loadFromStateDict(stateDicts[0])
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in range(1,len(stateDicts)):
            nextModel = ResNet.loadFromStateDict(stateDicts[s])
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
        model = ResNet.loadFromStateDict(torch.load(statePaths[0], map_location=lambda storage, loc: storage))
        if len(statePaths) == 1:
            return model
        
        pars = dict(model.named_parameters())
        for key in pars:
            pars[key] = pars[key].detach().cpu()
        for s in statePaths[1:]:
            model = ResNet.loadFromStateDict(torch.load(s, map_location=lambda storage, loc: storage))
            state = dict(model.named_parameters())
            for key in state:
                pars[key] += state[key]
        
        finalState = torch.load(statePaths[-1], map_location=lambda storage, loc: storage)
        for key in pars:
            finalState['model_state'][key] = pars[key] / (len(statePaths))
        
        model = ResNet.loadFromStateDict(finalState)
        return model


    def getParameters(self,freezeFE=False):
        headerParams = list(self.classifier.parameters())
        if freezeFE:
            return headerParams
        else:
            return list(self.fe.parameters()) + headerParams

    
    def forward(self, x, isFeatureVector=False):
        bs = x.size(0)
        if isFeatureVector:
            yhat = self.classifier(x.view(bs, -1))
        else:
            fVec = self.fe(x)
            yhat = self.classifier(fVec.view(bs, -1))
        return yhat