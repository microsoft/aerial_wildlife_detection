'''
    RetinaNet specifier for Detectron2 model trainer in AIDE.

    2020 Benjamin Kellenberger
'''

import json
import torch
from detectron2 import model_zoo

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.genericDetectronBBoxModel import GenericDetectron2BoundingBoxModel
from ai.models.detectron2.boundingBoxes.retinanet import DEFAULT_OPTIONS
from util import optionsHelper


class RetinaNet(GenericDetectron2BoundingBoxModel):

    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/boundingBoxes/retinanet.json',
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
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'RetinaNet', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a RetinaNet instance.'

        # modify model weights to accept new label classes
        if len(newClasses):

            weights = model.head.cls_score.weight    # a anchors x n classes
            biases = model.head.cls_score.bias

            numNeurons = len(biases)
            numClasses_orig = len(stateDict['labelclassMap'].keys()) - len(newClasses)
            numAnchors = numNeurons // numClasses_orig


            # create weights and biases for new classes
            if True:        #TODO: add flags in config file
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                modelClasses = range(model.num_classes)
                correlations = self.calculateClassCorrelations(model, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images

                correlations_expanded = correlations.repeat(1,numAnchors).to(weights.device)

                for cl in range(len(newClasses)):

                    #TODO: argmax test
                    bestMatch = correlations.argmax(1)[cl]

                    newWeight = weights_copy[cl*numAnchors:((cl+1)*numAnchors),...]
                    newBias = biases_copy[cl*numAnchors:((cl+1)*numAnchors)]

                    # #TODO: faulty correlation test... produces rubbish predictions
                    # newWeight = weights_copy * correlations_expanded[cl,:].view(-1,1,1,1)       #TODO: choose plan B for classes with no match
                    # newBias = biases_copy * correlations_expanded[cl,:]

                    # _, C, W, H = newWeight.size()
                    # newWeight = newWeight.view(numAnchors, -1, C, W, H)
                    # newBias = newBias.view(numAnchors, -1)
                    # corr = correlations_expanded[cl,:].view(numAnchors, -1)
                    # valid = torch.nonzero(corr, as_tuple=False)

                    # # average
                    # newWeight = newWeight.sum(1) / (newWeight>0).sum(1)
                    # newWeight[1 - torch.isfinite(newWeight).long()] = 0        #TODO

                    # newBias = newBias.sum(1) / (newBias>0).sum(1)
                    # newBias[1 - torch.isfinite(newBias).long()] = 0     #TODO

                    # prepend (last column is background class)
                    weights = torch.cat((newWeight, weights), 0)
                    biases = torch.cat((newBias, biases), 0)

            else:
                #TODO: suboptimal intermediate solution: find set of sum of weights and biases with minimal difference to zero
                massValues = []
                for idx in range(0, weights.size(0), numAnchors):
                    wbSum = torch.sum(torch.abs(weights[idx:(idx+numAnchors),...])) + \
                            torch.sum(torch.abs(biases[idx:(idx+numAnchors)]))
                    massValues.append(wbSum.unsqueeze(0))
                massValues = torch.cat(massValues, 0)
                
                smallest = torch.argmax(massValues)     #TODO

                newWeights = weights[(smallest*numAnchors):(smallest+1)*numAnchors,...]
                newBiases = biases[(smallest*numAnchors):(smallest+1)*numAnchors]

                for cl in newClasses:
                    # add a tiny bit of noise for better specialization capabilities (TODO: assess long-term effect of that...)
                    noiseW = 0.01 * (0.5 - torch.rand_like(newWeights))
                    noiseB = 0.01 * (0.5 - torch.rand_like(newBiases))
                    weights = torch.cat((newWeights.clone() + noiseW, weights), 0)
                    biases = torch.cat((newBiases.clone() + noiseB, biases), 0)
            
            # apply updated weights and biases

            #TODO: remove old
            weights = weights[:(len(newClasses)*numAnchors),...]
            biases = biases[:(len(newClasses)*numAnchors)]
            lcMap = {}
            for idx, lc in enumerate(newClasses):
                lcMap[lc] = idx
            stateDict['labelclassMap'] = lcMap

            model.head.cls_score.weight = torch.nn.Parameter(weights)
            model.head.cls_score.bias = torch.nn.Parameter(biases)
                
            print(f'Neurons for {len(newClasses)} new label classes added to RetinaNet model.')


        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.RETINANET.NUM_CLASSES = len(stateDict['labelclassMap'])
        model.num_classes = len(stateDict['labelclassMap'])
        return model, stateDict



#%%
#TODO: just for debugging; remove at final revision

if __name__ == '__main__':

    # meta data
    project = 'gzgcid'

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
    
    # get model settings
    queryStr = '''
        SELECT ai_model_library, ai_model_settings FROM aide_admin.project
        WHERE shortname = %s;
    '''
    result = database.execute(queryStr, (project,), 1)
    modelLibrary = result[0]['ai_model_library']
    modelSettings = result[0]['ai_model_settings']

    # launch model
    rn = RetinaNet(project, config, database, fileServer, modelSettings)
    # rn.update_model(None, data, updateStateFun)
    rn.train(None, data, updateStateFun)
    rn.inference(None, data, updateStateFun)