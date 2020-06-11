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


    def updateModel(self, labelClasses, addMissing=False, removeObsolete=False):
        '''
            Receives a new dict of label classes ("labelClasses") and compares
            it with the current one. If "labelClasses" contains new label classes
            that are not present in the current, and if "addMissing" is True, new
            neurons are added for each class. Likewise, if the current model predicts
            label classes that are not present in the new "labelClasses", and if
            "removeObsolete" is True, those neurons are being removed.
        '''
        if not addMissing or not removeObsolete:
            return
        
        classes_current = set([lc for lc in self.labelclassMap.keys()])
        classes_new = set([lc for lc in labelClasses.keys()])
        classes_missing = classes_new.difference(classes_current)
        classes_obsolete = classes_current.difference(classes_new)

        # add new neurons
        if addMissing and len(classes_missing):
            weights = self.classifier[-2].weight
            biases = self.classifier[-2].bias

            # find set of sum of weights and biases with minimal difference to zero
            massValues = []
            for idx in range(0, weights.size(0), self.numAnchors):
                wbSum = torch.sum(torch.abs(weights[idx:(idx+self.numAnchors),...])) + \
                        torch.sum(torch.abs(biases[idx:(idx+self.numAnchors)]))
                massValues.append(wbSum.unsqueeze(0))
            massValues = torch.cat(massValues, 0)
            
            smallest = torch.argmin(massValues)

            newWeights = weights[smallest:(smallest+1), ...]
            newBiases = biases[smallest:(smallest+1)]

            for classname in classes_missing:
                # add a tiny bit of noise for better specialization capabilities (TODO: assess long-term effect of that...)
                noiseW = 0.01 * (0.5 - torch.rand_like(newWeights))
                noiseB = 0.01 * (0.5 - torch.rand_like(newBiases))
                weights = torch.cat((weights, newWeights.clone() + noiseW), 0)
                biases = torch.cat((biases, newBiases.clone() + noiseB), 0)

                # update labelclass map
                self.labelclassMap[classname] = len(self.labelclassMap)
        
            # apply updated weights and biases
            self.classifier[-2].weight = nn.Parameter(weights)
            self.classifier[-2].bias = nn.Parameter(biases)

            print(f'Neurons for {len(classes_missing)} new label classes added to ResNet model.')

        # remove superfluous neurons
        if removeObsolete and len(classes_obsolete):
            weights = self.classifier[-2].weight
            biases = self.classifier[-2].bias

            for classname in classes_obsolete:
                classIdx = self.labelclassMap[classname]

                # remove neurons: slice tensors
                weights = torch.cat((weights[0:classIdx,...], weights[(classIdx+1):,...]), 0)
                biases = torch.cat((biases[0:classIdx], biases[(classIdx+1):]), 0)

                # shift down indices of labelclass map
                del self.labelclassMap[classname]
                for key in self.labelclassMap.keys():
                    if self.labelclassMap[key] > classIdx:
                        self.labelclassMap[key] -= 1

            # apply updated weights and biases
            self.classifier[-2].weight = nn.Parameter(weights)
            self.classifier[-2].bias = nn.Parameter(biases)

            print(f'Neurons of {len(classes_obsolete)} obsolete label classes removed from RetinaNet model.')

        self.numClasses = len(self.labelclassMap.keys())

    
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