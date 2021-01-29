'''
    2021 Benjamin Kellenberger
'''

import os
import torch
from detectron2.config import get_cfg

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.labels.genericDetectronLabelModel import GenericDetectron2LabelModel
from ai.models.detectron2.labels.torchvisionClassifier import TorchvisionClassifier as Model, DEFAULT_OPTIONS, add_torchvision_classifier_config
from util import optionsHelper


class TorchvisionClassifier(GenericDetectron2LabelModel):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(TorchvisionClassifier, self).__init__(project, config, dbConnector, fileServer, options)

        try:
            if self.detectron2cfg.MODEL.META_ARCHITECTURE != 'TorchvisionClassifier':
                # invalid options provided
                raise
        except:
            print(f'[{self.project}] WARNING: provided options are not valid for the Torchvision classifier model; falling back to defaults.')
            self.options = self.getDefaultOptions()
            self.detectron2cfg = self._get_config()
            self.detectron2cfg = GenericDetectron2Model.parse_aide_config(self.options, self.detectron2cfg)



    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/torchvisionclassifier.json',
            DEFAULT_OPTIONS
        )


    
    def _get_config(self):
        cfg = get_cfg()
        add_torchvision_classifier_config(cfg)
        defaultConfig = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'config', 'value'])
        if isinstance(defaultConfig, dict):
            defaultConfig = defaultConfig['id']
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
        '''
        model, stateDict, newClasses = self.initializeModel(stateDict, data)
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'TorchvisionClassifier', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a Torchvision classifier instance.'
        
        # modify model weights to accept new label classes
        if len(newClasses):

            # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            classificationLayer = Model.get_classification_layer(model, self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR)

            weights = classificationLayer.weight
            biases = classificationLayer.bias

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                modelClasses = range(len(biases))
                correlations = self.calculateClassCorrelations(model, modelClasses, newClasses, updateStateFun, 128)    #TODO: num images
                correlations = correlations.to(weights.device)

                classMatches = (correlations.sum(1) > 0)            #TODO: calculate alternative strategies (e.g. class name similarities)

                randomIdx = torch.randperm(len(biases)-1)
                if len(randomIdx) < len(newClasses):
                    # source model has fewer classes than target model; repeat
                    randomIdx = randomIdx.repeat(int(len(newClasses)/len(biases)+1))
                
                for cl in range(len(newClasses)):

                    if classMatches[cl].item():
                        newWeight = weights_copy.clone() * correlations[cl,:].unsqueeze(-1)
                        newBias = biases_copy.clone() * correlations[cl,:]

                        valid = (correlations[cl,:] > 0)

                        # average
                        newWeight = (newWeight.sum(0) / valid.sum()).unsqueeze(0)
                        newBias = newBias.sum() / valid.sum().unsqueeze(0)
                    
                    else:
                        # class has no match; use alternative solution (this should not happen with classification models)

                        #TODO: suboptimal alternative solution: choose random class
                        newWeight = weights_copy.clone()[randomIdx[cl],:].unsqueeze(0)
                        newBias = biases_copy.clone()[randomIdx[cl]].unsqueeze(0)

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
                if clName not in data['labelClasses']:
                    valid[idx] = 0
                else:
                    classMap_updated[clName] = index_updated
                    index_updated += 1

            weights = weights[valid,...]
            biases = biases[valid]

            # apply updated weights and biases
            classificationLayer.weight = torch.nn.Parameter(weights)
            classificationLayer.bias = torch.nn.Parameter(biases)
            if hasattr(classificationLayer, 'out_features'):
                classificationLayer.out_features = len(biases)
            Model.set_classification_layer(model, classificationLayer, self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR)
            stateDict['labelclassMap'] = classMap_updated

            print(f'[{self.project}] Neurons for {len(newClasses)} new label classes added to Torchvision classifier model.')

        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.TVCLASSIFIER.NUM_CLASSES = len(stateDict['labelclassMap'])
        return model, stateDict


#%%
#TODO: just for debugging; remove at final revision

if __name__ == '__main__':

    # meta data
    project = 'abcde'

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

    #TODO
    modelSettings = {
        'options': {
            'model': {
                'config': {
                    'value': 'labels/vgg16_ImageNet.yaml'
                }
            }
        }
    }

    # launch model
    rn = TorchvisionClassifier(project, config, database, fileServer, modelSettings)
    stateDict = rn.update_model(None, data, updateStateFun)
    stateDict, stats = rn.train(stateDict, data, updateStateFun)
    rn.inference(stateDict, data, updateStateFun)