'''
    Faster R-CNN specifier for Detectron2 model trainer in AIDE.

    2020 Benjamin Kellenberger
'''

import json
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.genericDetectronBBoxModel import GenericDetectron2BoundingBoxModel
from ai.models.detectron2.boundingBoxes.fasterrcnn import DEFAULT_OPTIONS
from util import optionsHelper


class FasterRCNN(GenericDetectron2BoundingBoxModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(FasterRCNN, self).__init__(project, config, dbConnector, fileServer, options)

        try:
            if self.detectron2cfg.MODEL.META_ARCHITECTURE != 'GeneralizedRCNN':
                # invalid options provided
                raise
        except:
            print('WARNING: provided options are not valid for Faster R-CNN; falling back to defaults.')
            self.options = self.getDefaultOptions()
            self.detectron2cfg = self._get_config()
            self.detectron2cfg = GenericDetectron2Model.parse_aide_config(self.options, self.detectron2cfg)

        #TODO: Faster R-CNN currently does not work with empty images
        self.detectron2cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True



    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/boundingBoxes/fasterrcnn.json',
            DEFAULT_OPTIONS
        )



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
        model, stateDict, newClasses = self.initializeModel(stateDict, data)
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a Faster R-CNN instance.'

        # modify model weights to accept new label classes      #TODO: current implementation below causes loss to go to infinity...
        if len(newClasses):

            # class predictor parameters
            class_weights = model.roi_heads.box_predictor.cls_score.weight
            class_biases = model.roi_heads.box_predictor.cls_score.bias

            # box predictor parameters
            bbox_weights = model.roi_heads.box_predictor.bbox_pred.weight
            bbox_biases = model.roi_heads.box_predictor.bbox_pred.bias


            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                class_weights_copy = class_weights.clone()
                class_biases_copy = class_biases.clone()
                bbox_weights_copy = bbox_weights.clone()
                bbox_biases_copy = bbox_biases.clone()

                modelClasses = range(len(class_biases))
                correlations = self.calculateClassCorrelations(model, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images
                correlations = correlations[:,:-1].to(class_weights.device)      # exclude background class

                classMatches = (correlations.sum(1) > 0)            #TODO: calculate alternative strategies (e.g. class name similarities)

                randomIdx = torch.randperm(len(class_biases)-1)
                if len(randomIdx) < len(newClasses):
                    # source model has fewer classes than target model; repeat
                    randomIdx = randomIdx.repeat(int(len(newClasses)/len(class_biases)+1))

                for cl in range(len(newClasses)):

                    if classMatches[cl].item():
                        newClassWeight = class_weights_copy.clone()[:-1,:] * correlations[cl,:].unsqueeze(-1)
                        newClassBias = class_biases_copy.clone()[:-1] * correlations[cl,:]
                        newBoxWeight = bbox_weights_copy.clone() * correlations[cl,:].unsqueeze(-1).repeat(4,1)
                        newBoxBias = bbox_biases_copy.clone() * correlations[cl,:].repeat(4)

                        valid = (correlations[cl,:] > 0)

                        # average
                        newClassWeight = (newClassWeight.sum(0) / valid.sum()).unsqueeze(0)
                        newClassBias = newClassBias.sum() / valid.sum().unsqueeze(0)
                        newBoxWeight = newBoxWeight.view(-1, 4, bbox_weights_copy.size(-1)).sum(0) / valid.sum()
                        newBoxBias = newBoxBias.view(-1, 4).sum(0) / valid.sum()

                    else:
                        # class has no match; use alternative solution

                        #TODO: suboptimal alternative solution: choose random class
                        newClassWeight = class_weights_copy.clone()[randomIdx[cl],:].unsqueeze(0)
                        newClassBias = class_biases_copy.clone()[randomIdx[cl]].unsqueeze(0)
                        newBoxWeight = bbox_weights_copy.clone()[randomIdx[cl]*4:(randomIdx[cl]+1)*4,:]
                        newBoxBias = bbox_biases_copy.clone()[randomIdx[cl]*4:(randomIdx[cl]+1)*4]

                        # add a bit of noise
                        newClassWeight += (0.5 - torch.rand_like(newClassWeight)) * 0.5 * torch.std(class_weights_copy)
                        newClassBias += (0.5 - torch.rand_like(newClassBias)) * 0.5 * torch.std(class_biases_copy)
                        newBoxWeight += (0.5 - torch.rand_like(newBoxWeight)) * 0.5 * torch.std(bbox_weights_copy)
                        newBoxBias += (0.5 - torch.rand_like(newBoxBias)) * 0.5 * torch.std(bbox_biases_copy)

                    # prepend (last column is background class)
                    class_weights = torch.cat((newClassWeight, class_weights), 0)
                    class_biases = torch.cat((newClassBias, class_biases), 0)
                    bbox_weights = torch.cat((newBoxWeight, bbox_weights), 0)
                    bbox_biases = torch.cat((newBoxBias, bbox_biases), 0)

            # remove old classes
            classmap_updated = {}
            valid_cls = torch.ones(len(class_biases), dtype=torch.bool)
            valid_box = torch.ones(len(bbox_biases), dtype=torch.bool)
            classmap_inv = {}
            for key in stateDict['labelclassMap'].keys():
                classmap_inv[stateDict['labelclassMap'][key]] = key
            index_updated = 0
            for clIdx in range(len(classmap_inv)):
                clName = classmap_inv[clIdx]
                if clName not in data['labelClasses']:
                    index = clIdx + len(newClasses)   # we prepended new class weights; need to add offset
                    valid_cls[index] = False
                    valid_box[(index*4):(index+1)*4] = False
                else:
                    classmap_updated[clName] = index_updated
                    index_updated += 1
            stateDict['labelclassMap'] = classmap_updated
            class_weights = class_weights[valid_cls,:]
            class_biases = class_biases[valid_cls]
            bbox_weights = bbox_weights[valid_box,:]
            bbox_biases = bbox_biases[valid_box]

            # apply updated weights and biases
            model.roi_heads.box_predictor.cls_score.weight = torch.nn.Parameter(class_weights)
            model.roi_heads.box_predictor.cls_score.bias = torch.nn.Parameter(class_biases)

            model.roi_heads.box_predictor.bbox_pred.weight = torch.nn.Parameter(bbox_weights)
            model.roi_heads.box_predictor.bbox_pred.bias = torch.nn.Parameter(bbox_biases)

            print(f'Neurons for {len(newClasses)} new label classes added to Faster R-CNN model.')
        
        #TODO: remove superfluous?

        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.ROI_HEADS.NUM_CLASSES = len(stateDict['labelclassMap'])
        model.roi_heads.box_predictor.cls_score.out_features = len(stateDict['labelclassMap'])
        model.roi_heads.box_predictor.bbox_pred.out_features = len(stateDict['labelclassMap'])
        return model, stateDict



#%%
#TODO: just for debugging; remove at final revision

if __name__ == '__main__':

    from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

    # meta data
    project = 'aerialelephants_wc'

    # set up parts of AIDE
    import os
    os.environ['AIDE_CONFIG_PATH'] = 'settings_multiProject.ini'
    os.environ['AIDE_MODULES'] = ''

    from util.configDef import Config
    from modules.AIController.backend.functional import AIControllerWorker
    from modules.AIWorker.backend.fileserver import FileServer
    from modules.AIWorker.backend.worker.functional import __load_metadata
    from modules.Database.app import Database

    config = Config()
    fileServer = FileServer(config).get_secure_instance(project)
    database = Database(config)

    aicw = AIControllerWorker(config, None)
    data = aicw.get_training_images(
        project=project,
        maxNumImages=512)
    data = __load_metadata(project, database, data[0], True)

    def updateStateFun(state, message, done=None, total=None):
        print(f'{message}: {done}/{total}')
    

    # launch model
    rn = FasterRCNN(project, config, database, fileServer, None)
    stateDict = rn.update_model(None, data, updateStateFun)
    rn.inference(stateDict, data, updateStateFun)