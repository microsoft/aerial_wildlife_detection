'''
    DeepForest specifier for Detectron2 model trainer in AIDE:
    https://github.com/weecology/DeepForest

    2022 Benjamin Kellenberger
'''

import os
import torch
from detectron2.config import get_cfg
from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.genericDetectronBBoxModel import GenericDetectron2BoundingBoxModel
from ai.models.detectron2.boundingBoxes.deepforest import DEFAULT_OPTIONS, deepforest_parts
from util import optionsHelper


class DeepForest(GenericDetectron2BoundingBoxModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(DeepForest, self).__init__(project, config, dbConnector, fileServer, options)

        try:
            if self.detectron2cfg.MODEL.META_ARCHITECTURE != 'DeepForest':
                # invalid options provided
                raise Exception('Invalid model architecture')
        except:
            print('WARNING: provided options are not valid for DeepForest; falling back to defaults.')
            self.options = self.getDefaultOptions()
            self.detectron2cfg = self._get_config()
            self.detectron2cfg = GenericDetectron2Model.parse_aide_config(self.options, self.detectron2cfg)


    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/boundingBoxes/deepforest.json',
            DEFAULT_OPTIONS
        )


    @staticmethod
    def _add_deepforest_config(cfg):
        cfg.set_new_allowed(True)
        baseConfigFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs/boundingBoxes/deepforest/deepforest.yaml')
        cfg.merge_from_file(baseConfigFile)

        cfg.MODEL.META_ARCHITECTURE = 'DeepForest'
        cfg.MODEL.BACKBONE.NAME = 'DeepForestBackbone'
        cfg.MODEL.DEEPFOREST_PRETRAINED = 'deepforest'

    
    def _get_config(self):
        cfg = get_cfg()
        self._add_deepforest_config(cfg)
        defaultConfig = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'config', 'value', 'id'])
        configFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs', defaultConfig)
        cfg.merge_from_file(configFile)
        return cfg

    
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
        model, stateDict, newClasses, projectToStateMap = self.initializeModel(stateDict, data)
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'DeepForest', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a DeepForest instance.'

        # modify model weights to accept new label classes
        if len(newClasses):
            
            # create temporary labelclassMap for new classes
            lcMap_new = dict(zip(newClasses, list(range(len(newClasses)))))

             # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            cls_layer = model.model.head.classification_head.cls_logits        # layers with n_features x (n_anchors x n_classes)
            
            numNeurons = len(cls_layer.bias)
            numAnchors = len(model.model.head.regression_head.bbox_reg.bias) // 4
            numClasses_model = numNeurons // numAnchors

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                modelClasses = range(numClasses_model)
                correlations = self.calculateClassCorrelations(stateDict, model, lcMap_new, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images

                existingIdx = torch.tensor(range(0, numNeurons, numAnchors))        # starting indices of existing class weights
                randomIdx = torch.randperm(len(existingIdx))
                if len(randomIdx) < len(newClasses):
                    # source model has fewer classes than target model; repeat
                    randomIdx = randomIdx.repeat(int(len(newClasses)/len(randomIdx)+1))     #TODO

                weights = cls_layer.weight
                biases = cls_layer.bias

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

                    classVector.insert(0, newClasses[cl])
            
                    # apply updated weights and biases
                    model.model.head.classification_head.cls_logits.weight = torch.nn.Parameter(weights)
                    model.model.head.classification_head.cls_logits.bias = torch.nn.Parameter(biases)
                    model.model.head.classification_head.cls_logits.out_channels = len(biases)

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

            print(f'Neurons for {len(newClasses)} new label classes added to DeepForest model.')


        # finally, update model and config
        if len(stateDict['labelclassMap']):
            stateDict['detectron2cfg'].MODEL.RETINANET.NUM_CLASSES = len(stateDict['labelclassMap'])
            map_inv = dict([v,k] for k,v in stateDict['labelclassMap'].items())
            mapKeys = list(map_inv)
            mapKeys.sort()
            model.names = [str(map_inv[key]) for key in mapKeys]
        stateDict['detectron2cfg'].MODEL.DEEPFOREST_PRETRAINED = False
        return model, stateDict