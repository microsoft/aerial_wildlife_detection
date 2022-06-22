'''
    U-net specifier for Detectron2 model trainer in AIDE.

    2022 Benjamin Kellenberger
'''

import os
import math
import torch
from detectron2.config import get_cfg

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.segmentationMasks.genericDetectronSegmentationModel import GenericDetectron2SegmentationModel
from ai.models.detectron2.segmentationMasks.unet import DEFAULT_OPTIONS, unet_parts
from util import optionsHelper


class Unet(GenericDetectron2SegmentationModel):

    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/segmentationMasks/unet.json',
            DEFAULT_OPTIONS
        )

    
    @staticmethod
    def _add_unet_config(cfg):
        cfg.MODEL.BACKBONE.NAME = 'UnetEncoder'
        cfg.MODEL.BACKBONE.NUM_CHANNELS = 3
        cfg.MODEL.SEM_SEG_HEAD.NAME = 'UnetDecoder'
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
        cfg.MODEL.BILINEAR = True


    def _get_config(self):
        cfg = get_cfg()
        self._add_unet_config(cfg)
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
        model, stateDict, newClasses, _ = self.initializeModel(stateDict, data)
        # modify model weights to accept new label classes
        if len(newClasses):

            # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            weights = model.sem_seg_head.outc.conv.weight
            biases = model.sem_seg_head.outc.conv.bias
            numClasses_orig = len(biases)

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                #TODO: we currently have no indexing possibilities to retrieve images with correct labels...
                # correlations = self.calculateClassCorrelations(stateDict, model, lcMap_new, range(numClasses_orig), newClasses, updateStateFun, 128)    #TODO: num images
                
                # use alternative solution: choose random class
                randomOrder = torch.randperm(numClasses_orig).repeat(max(1, int(math.ceil(len(newClasses)/float(numClasses_orig)))))
                for cl in range(len(newClasses)):
                    newWeight = weights_copy[randomOrder[cl],...]
                    newBias = biases_copy[randomOrder[cl]]

                    # add a bit of noise
                    newWeight += (0.5 - torch.rand_like(newWeight)) * 0.5 * torch.std(weights_copy)
                    newBias += (0.5 - torch.rand_like(newBias)) * 0.5 * torch.std(biases_copy)

                    # prepend
                    weights = torch.cat((newWeight.unsqueeze(0), weights), 0)
                    biases = torch.cat((newBias.unsqueeze(0), biases), 0)
                    classVector.insert(0, newClasses[cl])

            # remove old classes
            # valid = torch.ones(len(biases), dtype=torch.bool)
            classMap_updated = {}
            index_updated = 0
            for idx, clName in enumerate(classVector):
                # if clName not in data['labelClasses']:
                #     valid[idx] = 0
                # else:
                if True:    # we don't remove old classes anymore (TODO: flag in configuration)
                    classMap_updated[clName] = index_updated
                    index_updated += 1

            # weights = weights[valid,...]
            # biases = biases[valid,...]

            # apply updated weights and biases
            model.sem_seg_head.outc.conv.weight = torch.nn.Parameter(weights)
            model.sem_seg_head.outc.conv.bias = torch.nn.Parameter(biases)
            model.sem_seg_head.outc.conv.out_channels = len(biases)

            stateDict['labelclassMap'] = classMap_updated
                
            print(f'Neurons for {len(newClasses)} new label classes added to U-net model.')

        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(stateDict['labelclassMap'])
        model.num_classes = len(stateDict['labelclassMap'])
        return model, stateDict