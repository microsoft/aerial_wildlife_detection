'''
    YOLOv5 specifier for Detectron2 model trainer in AIDE.

    2022 Benjamin Kellenberger
'''

import os
import torch
from detectron2.config import get_cfg
from detectron2.data import transforms as T
import yolov5
from yolov5.models.experimental import attempt_load
import sys
sys.path.insert(0, sys.modules['yolov5'].__path__[0])  # required to be able to load entire models

from ai.models.detectron2._functional.checkpointer import DetectionCheckpointerInMem
from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.genericDetectronBBoxModel import GenericDetectron2BoundingBoxModel
from ai.models.detectron2.boundingBoxes.yolov5 import DEFAULT_OPTIONS, yolov5_model
from util import optionsHelper


class YOLOv5(GenericDetectron2BoundingBoxModel):

    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/boundingBoxes/yolov5.json',
            DEFAULT_OPTIONS
        )

    
    @staticmethod
    def _add_yolov5_config(cfg):
        cfg.set_new_allowed(True)
        baseConfigFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs/boundingBoxes/yolov5/base-yolov5.yaml')
        cfg.merge_from_file(baseConfigFile)

        cfg.INPUT.IMAGE_SIZE = 640
        #TODO: from train.py script:
        # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        # imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        cfg.MODEL.META_ARCHITECTURE = 'YOLOv5'
        cfg.MODEL.TEST_TIME_AUGMENT = False
        cfg.MODEL.NUM_CHANNELS = 3
        cfg.MODEL.NUM_CLASSES = 80  # MS-COCO default
    

    def _get_config(self):
        cfg = get_cfg()
        self._add_yolov5_config(cfg)
        defaultConfig = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'config', 'value', 'id'])
        configFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs', defaultConfig)
        cfg.merge_from_file(configFile)
        return cfg
    

    def initializeTransforms(self, mode='train', options=None):
        '''
            Override of the superclass method to ensure image resize and
            normalization transforms are always there.
        '''
        transforms = super(YOLOv5, self).initializeTransforms(mode, options)
        if T.Resize not in transforms:
            transforms.append(T.Resize(shape=(self.detectron2cfg.INPUT.IMAGE_SIZE, self.detectron2cfg.INPUT.IMAGE_SIZE)))
        return transforms


    def loadModelWeights(self, model, stateDict, forceNewModel):
        '''
            Override of superclass method to be able to load YOLOv5 models from
            either a *.pt checkpoint or torch.hub.
        '''
        if 'model' in stateDict and not forceNewModel:
            # trained weights available
            checkpointer = DetectionCheckpointerInMem(model)
            checkpointer.loadFromObject(stateDict)
        else:
            # fresh model; initialize from YOLOv5 spec
            weightPath = self.detectron2cfg.MODEL.WEIGHTS
            if len(weightPath):
                #TODO: check if torch.hub
                model_ = attempt_load(weightPath)       # YOLOv5 saves entire models...
                model.load_weights(model_.state_dict(), strict=False)
                #TODO: this overrides user-specified inputs; better provide in config file within AIDE
                # if hasattr(model_, 'hyp'):
                #     model.model.hyp = model_.hyp


    def loadAndAdaptModel(self, stateDict, data, updateStateFun):
        '''
            Loads a model and a labelclass map from a given "stateDict".
            First calls the parent implementation to obtain a default
            model, then checks for new label classes and modifies the
            model's classification head accordingly.
            TODO: implement advanced modifiers:
            1. Weighted linear combination of images with new annotations
               present
            2. Weighted linear combination according to similarity of name
               of new classes w.r.t. existing ones (e.g. using Word2Vec)
            
            For now, only the smallest existing class weights are used
            and duplicated.
        '''
        model, stateDict, newClasses, _ = self.initializeModel(stateDict, data)
        # modify model weights to accept new label classes
        if True:       #TODO: adaptlen(newClasses):
            
            # create temporary labelclassMap for new classes
            lcMap_new = dict(zip(newClasses, list(range(len(newClasses)))))

            # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            layers = model.model.model[-1].m        # layers with a anchors x (n classes + 5 outputs) #TODO: check if this applies to all YOLOv5 model sizes
            
            numNeurons = len(layers[0].bias)
            numAnchors = len(model.model.yaml['anchors'])
            numClasses_model = len(model.model.names)

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                modelClasses = range(len(model.model.names))
                correlations = self.calculateClassCorrelations(stateDict, model, lcMap_new, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images

                existingIdx = torch.tensor(range(0, numNeurons-(numAnchors*5), numAnchors))        # starting indices of existing class weights
                randomIdx = torch.randperm(len(existingIdx))
                if len(randomIdx) < len(newClasses):
                    # source model has fewer classes than target model; repeat
                    randomIdx = randomIdx.repeat(int(len(newClasses)/len(randomIdx)+1))     #TODO

                for lIdx, l in enumerate(range(len(layers))):
                    weights = layers[l].weight
                    biases = layers[l].bias

                    weights_copy = weights.clone()
                    biases_copy = biases.clone()

                    classMatches = (correlations.sum(1) > 0)            #TODO: calculate alternative strategies (e.g. class name similarities)

                    for cl in range(len(newClasses)):

                        if classMatches[cl].item():
                            #TODO: not correct yet
                            newWeight = weights_copy[:numClasses_model*numAnchors,...]
                            newBias = biases_copy[:numClasses_model*numAnchors]

                            corr = correlations[cl,:]
                            valid = (corr > 0)

                            # average
                            newWeight = torch.zeros((5, *weights_copy.size()[1:]), dtype=weights_copy.dtype, device=weights_copy.device)
                            newBias = torch.zeros((5,), dtype=biases_copy.dtype, device=biases_copy.device)
                            for v in torch.where(valid)[0]:
                                newWeight += weights_copy[v*numAnchors:(v+1)*numAnchors,...]
                                newBias += biases_copy[v*numAnchors:(v+1)*numAnchors]
                            
                            newWeight /= corr[valid].sum()
                            newBias /= corr[valid].sum()
                        
                        else:
                            # class has no match; use alternative solution

                            #TODO: suboptimal alternative solution: choose random class
                            newWeight = weights_copy.clone()
                            newBias = biases_copy.clone()
                            idx = existingIdx[randomIdx[cl]]
                            newWeight = newWeight[idx:idx+numAnchors,...]
                            newBias = newBias[idx:idx+numAnchors]

                            # add a bit of noise
                            newWeight += (0.5 - torch.rand_like(newWeight)) * 0.5 * torch.std(weights_copy)
                            newBias += (0.5 - torch.rand_like(newBias)) * 0.5 * torch.std(biases_copy)

                        # prepend
                        weights = torch.cat((newWeight, weights), 0)
                        biases = torch.cat((newBias, biases), 0)

                        if lIdx == 0:
                            classVector.insert(0, newClasses[cl])
            
                    # apply updated weights and biases
                    model.model.model[-1].m[lIdx].weight = torch.nn.Parameter(weights)
                    model.model.model[-1].m[lIdx].bias = torch.nn.Parameter(biases)
                    model.model.model[-1].m[lIdx].out_channels = len(biases)

            # valid = torch.ones(len(biases), dtype=torch.bool)
            classMap_updated = {}
            index_updated = 0
            for idx, clName in enumerate(classVector):
                # if clName not in data['labelClasses']:
                #     valid[idx*numAnchors:(idx+1)*numAnchors] = 0
                # else:
                if True:    # we don't remove old classes anymore (TODO: flag in configuration)
                    classMap_updated[clName] = index_updated
                    index_updated += 1

            # weights = weights[valid,...]
            # biases = biases[valid,...]
            
            stateDict['labelclassMap'] = classMap_updated

            print(f'Neurons for {len(newClasses)} new label classes added to YOLOv5 model.')


        # finally, update model and config
        if len(stateDict['labelclassMap']):
            stateDict['detectron2cfg'].MODEL.NUM_CLASSES = len(stateDict['labelclassMap'])
            map_inv = dict([v,k] for k,v in stateDict['labelclassMap'].items())
            mapKeys = list(map_inv)
            mapKeys.sort()
            model.names = [str(map_inv[key]) for key in mapKeys]
            model.model.names = model.names
        return model, stateDict