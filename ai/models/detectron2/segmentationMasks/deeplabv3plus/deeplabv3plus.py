'''
    DeepLabV3+ specifier for Detectron2 model trainer in AIDE.

    2020-21 Benjamin Kellenberger
'''

import os

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
            'config/ai/model/detectron2/segmentationMasks/deeplabV3+.json',
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

            weights = model.sem_seg_head.predictor.weight
            biases = model.sem_seg_head.predictor.bias
            numClasses_orig = len(biases)

            # create weights and biases for new classes
            if True:        #TODO: add flags in config file about strategy
                weights_copy = weights.clone()
                biases_copy = biases.clone()

                correlations = self.calculateClassCorrelations(model, range(numClasses_orig), newClasses, updateStateFun, 128)    #TODO: num images
                print('debug')

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
    rn.train(stateDict, data, updateStateFun)