'''
    RetinaNet specifier for Detectron2 model trainer in AIDE.

    2020-23 Benjamin Kellenberger
'''

import torch

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.boundingBoxes.genericDetectronBBoxModel import GenericDetectron2BoundingBoxModel
from ai.models.detectron2.boundingBoxes.retinanet import DEFAULT_OPTIONS


class RetinaNet(GenericDetectron2BoundingBoxModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(RetinaNet, self).__init__(project, config, dbConnector, fileServer, options)

        try:
            if self.detectron2cfg.MODEL.META_ARCHITECTURE != 'RetinaNet':
                # invalid options provided
                raise Exception('Invalid model architecture')
        except Exception:
            print('WARNING: provided options are not valid for RetinaNet; falling back to defaults.')
            self.options = self.getDefaultOptions()
            self.detectron2cfg = self._get_config()
            self.detectron2cfg = GenericDetectron2Model.parse_aide_config(self.options, self.detectron2cfg)



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
        model, stateDict, newClasses, projectToStateMap = self.initializeModel(stateDict, data)
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'RetinaNet', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a RetinaNet instance.'

        # modify model weights to accept new label classes
        if len(newClasses):

            # create temporary labelclassMap for new classes
            lcMap_new = dict(zip(newClasses, list(range(len(newClasses)))))

            # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            weights = model.head.cls_score.weight    # a anchors x n classes
            biases = model.head.cls_score.bias

            numNeurons = len(biases)
            numClasses_orig = len(stateDict['labelclassMap'].keys())    #TODO: figure out why this was added: - len(newClasses)
            numAnchors = numNeurons // numClasses_orig


            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                modelClasses = range(model.num_classes)
                correlations = self.calculateClassCorrelations(stateDict, model, lcMap_new, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images
                correlations_expanded = correlations.repeat(1,numAnchors).to(weights.device)

                classMatches = (correlations.sum(1) > 0)            #TODO: calculate alternative strategies (e.g. class name similarities)

                randomIdx = torch.randperm(int(len(biases_copy)/numAnchors))
                if len(randomIdx) < len(newClasses):
                    # source model has fewer classes than target model; repeat
                    randomIdx = randomIdx.repeat(int(len(newClasses)+1))

                for cl in range(len(newClasses)):

                    if classMatches[cl].item():
                        newWeight = weights_copy * correlations_expanded[cl,:].view(-1,1,1,1)
                        newBias = biases_copy * correlations_expanded[cl,:]

                        _, C, W, H = newWeight.size()
                        newWeight = newWeight.view(numAnchors, -1, C, W, H)
                        newBias = newBias.view(numAnchors, -1)
                        corr = correlations_expanded[cl,:].view(numAnchors, -1)
                        valid = (corr > 0)

                        # average
                        newWeight = newWeight.sum(1) / valid.sum(1).view(-1, 1, 1, 1)
                        newBias = newBias.sum(1) / valid.sum(1)
                    
                    else:
                        # class has no match; use alternative solution

                        #TODO: suboptimal alternative solution: choose random class
                        _, C, W, H = weights_copy.size()
                        newWeight = weights_copy.clone().view(numAnchors, -1, C, W, H)
                        newBias = biases_copy.clone().view(numAnchors, -1)

                        newWeight = newWeight[:,randomIdx[cl],...]
                        newBias = newBias[:,randomIdx[cl]]

                        # add a bit of noise
                        newWeight += (0.5 - torch.rand_like(newWeight)) * 0.5 * torch.std(weights_copy)
                        newBias += (0.5 - torch.rand_like(newBias)) * 0.5 * torch.std(biases_copy)

                    # prepend
                    weights = torch.cat((newWeight, weights), 0)
                    biases = torch.cat((newBias, biases), 0)
                    classVector.insert(0, newClasses[cl])
            
            # remove old classes
            valid = torch.ones(len(biases), dtype=torch.bool)
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

            # apply updated weights and biases
            model.head.cls_score.weight = torch.nn.Parameter(weights)
            model.head.cls_score.bias = torch.nn.Parameter(biases)
            
            stateDict['labelclassMap'] = classMap_updated

            print(f'Neurons for {len(newClasses)} new label classes added to RetinaNet model.')


        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.RETINANET.NUM_CLASSES = len(stateDict['labelclassMap'])
        model.num_classes = len(stateDict['labelclassMap'])
        return model, stateDict