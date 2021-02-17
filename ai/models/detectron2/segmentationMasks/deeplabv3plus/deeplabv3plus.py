'''
    DeepLabV3+ specifier for Detectron2 model trainer in AIDE.

    2020-21 Benjamin Kellenberger
'''

import os
import torch
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.segmentationMasks.genericDetectronSegmentationModel import GenericDetectron2SegmentationModel
from ai.models.detectron2.segmentationMasks.deeplabv3plus import DEFAULT_OPTIONS
from util import optionsHelper


class DeepLabV3Plus(GenericDetectron2SegmentationModel):

    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/segmentationMasks/deeplabv3plus.json',
            DEFAULT_OPTIONS
        )



    def _get_config(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        defaultConfig = optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'config', 'value', 'id'])
        configFile = os.path.join(os.getcwd(), 'ai/models/detectron2/_functional/configs', defaultConfig)
        cfg.merge_from_file(configFile)

        # disable SyncBatchNorm if not running on distributed system
        if comm.get_world_size() <= 1:
            cfg.MODEL.RESNETS.NORM = 'BN'
            cfg.MODEL.SEM_SEG_HEAD.NORM = 'BN'

        return cfg



    def _build_lr_scheduler(self, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)



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
        
        # modify model weights to accept new label classes
        if len(newClasses):

            # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key

            weights = model.sem_seg_head.predictor.weight
            biases = model.sem_seg_head.predictor.bias
            numClasses_orig = len(biases)

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                #TODO: we currently have no indexing possibilities to retrieve images with correct labels...
                # correlations = self.calculateClassCorrelations(model, range(numClasses_orig), newClasses, updateStateFun, 128)    #TODO: num images
                
                # use alternative solution: choose random class
                randomOrder = torch.randperm(numClasses_orig)
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
            biases = biases[valid,...]

            # apply updated weights and biases
            model.sem_seg_head.predictor.weight = torch.nn.Parameter(weights)
            model.sem_seg_head.predictor.bias = torch.nn.Parameter(biases)

            stateDict['labelclassMap'] = classMap_updated
                
            print(f'Neurons for {len(newClasses)} new label classes added to DeepLabV3+ model.')

        # finally, update model and config
        stateDict['detectron2cfg'].MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(stateDict['labelclassMap'])
        model.num_classes = len(stateDict['labelclassMap'])
        return model, stateDict




#%%
#TODO: just for debugging; remove at final revision

if __name__ == '__main__':

    # meta data
    project = 'test-segmentation'

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
    rn = DeepLabV3Plus(project, config, database, fileServer, None)
    _, stateDict = rn.loadAndAdaptModel(None, data, updateStateFun)
    # stateDict = rn.train(stateDict, data, updateStateFun)
    rn.inference(stateDict, data, updateStateFun)