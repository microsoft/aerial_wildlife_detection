'''
    Faster R-CNN specifier for Detectron2 model trainer in AIDE.

    2020 Benjamin Kellenberger
'''

import json
import torch
from detectron2 import model_zoo

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.frcnn import DEFAULT_OPTIONS
from util import optionsHelper


class FasterRCNN(GenericDetectron2Model):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(FasterRCNN, self).__init__(project, config, dbConnector, fileServer, options)

        #TODO: Faster R-CNN currently does not work with empty images
        self.detectron2cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True



    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/boundingBoxes/frcnn.json',
            DEFAULT_OPTIONS
        )



    def loadAndAdaptModel(self, stateDict, data):
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
            weights = model.roi_heads.box_predictor.cls_score.weight
            biases = model.roi_heads.box_predictor.cls_score.bias

            # box predictor parameters
            bbox_weight = model.roi_heads.box_predictor.bbox_pred.weight
            bbox_biases = model.roi_heads.box_predictor.bbox_pred.bias
            
            #TODO: suboptimal intermediate solution: find set of sum of weights and biases with minimal difference to zero
            massValues = []
            for idx in range(0, weights.size(0)-1):     # -1 because last index is background class
                wbSum = torch.sum(torch.abs(weights[idx,...])) + \
                        torch.sum(torch.abs(biases[idx]))
                massValues.append(wbSum.unsqueeze(0))
            massValues = torch.cat(massValues, 0)
            
            smallest = torch.argmin(massValues)

            newWeights = weights[smallest,...].unsqueeze(0)
            newBiases = biases[smallest].unsqueeze(0)

            bbox_idx = 4*smallest
            newWeights_bbox = bbox_weight[bbox_idx:(bbox_idx+4),...]
            newBiases_bbox = bbox_biases[bbox_idx:(bbox_idx+4),...]

            for cl in newClasses:
                # add a tiny bit of noise for better specialization capabilities (TODO: assess long-term effect of that...)
                noiseW = 0.01 * (0.5 - torch.rand_like(newWeights))
                noiseB = 0.01 * (0.5 - torch.rand_like(newBiases))
                weights = torch.cat((weights, newWeights.clone() + noiseW), 0)
                biases = torch.cat((biases, newBiases.clone() + noiseB), 0)

                bbox_weight = torch.cat((bbox_weight, newWeights_bbox.clone()), 0)
                bbox_biases = torch.cat((bbox_biases, newBiases_bbox.clone()), 0)
            
            # apply updated weights and biases
            model.roi_heads.box_predictor.cls_score.weight = torch.nn.Parameter(weights)
            model.roi_heads.box_predictor.cls_score.bias = torch.nn.Parameter(biases)

            model.roi_heads.box_predictor.bbox_pred.weight = torch.nn.Parameter(bbox_weight)
            model.roi_heads.box_predictor.bbox_pred.bias = torch.nn.Parameter(bbox_biases)

            print(f'Neurons for {len(newClasses)} new label classes added to RetinaNet model.')
        
        #TODO: remove superfluous?

        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.RETINANET.NUM_CLASSES = len(stateDict['labelclassMap'])
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

    def updateStateFun(state, message, done, total):
        print(f'{message}: {done}/{total}')
    

    # launch model
    rn = FasterRCNN(project, config, database, fileServer, None)
    rn.inference(None, data, updateStateFun)