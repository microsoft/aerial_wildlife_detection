'''
    Mask R-CNN specifier for Detectron2 model trainer in AIDE.

    2022 Benjamin Kellenberger
'''

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.instanceSegmentation.maskrcnn import DEFAULT_OPTIONS


class MaskRCNN(GenericDetectron2Model):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(MaskRCNN, self).__init__(project, config, dbConnector, fileServer, options)

        try:
            if self.detectron2cfg.MODEL.META_ARCHITECTURE != 'GeneralizedRCNN':
                # invalid options provided
                raise Exception(f'Invalid meta architecture ("{self.detectron2cfg.MODEL.META_ARCHITECTURE}" != "GeneralizedRCNN")')
        except Exception as e:
            print(f'[{self.project}] WARNING: provided options are not valid for Mask R-CNN (message: "{str(e)}"); falling back to defaults.')
            self.options = self.getDefaultOptions()
            self.detectron2cfg = self._get_config()
            self.detectron2cfg = GenericDetectron2Model.parse_aide_config(self.options, self.detectron2cfg)

        # Mask R-CNN does not work with empty images
        self.detectron2cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    

    @classmethod
    def getDefaultOptions(cls):
        return GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/instanceSegmentation/maskrcnn.json',
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
        assert self.detectron2cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNN', \
            f'ERROR: model meta-architecture "{self.detectron2cfg.MODEL.META_ARCHITECTURE}" is not a Faster R-CNN instance.'

        # modify model weights to accept new label classes
        if len(newClasses):
            #TODO
            lcMap_new = dict(zip(newClasses, list(range(len(newClasses)))))

             # create vector of label classes
            classVector = len(stateDict['labelclassMap']) * [None]
            for (key, index) in zip(stateDict['labelclassMap'].keys(), stateDict['labelclassMap'].values()):
                classVector[index] = key
            classMap_updated = {}
            index_updated = len(newClasses)
            for idx, clName in enumerate(classVector):
                # if clName not in data['labelClasses']:
                #     valid_cls[idx] = 0
                #     valid_box[(idx*4):(idx+1)*4] = 0
                # else:
                if True:    # we don't remove old classes anymore (TODO: flag in configuration)
                    classMap_updated[clName] = index_updated
                    index_updated += 1
                if idx >= len(classVector) - len(newClasses):
                    break
            
            #TODO
            index_updated = 0
            for lc in newClasses:
                classMap_updated[lc] = index_updated
                index_updated += 1
            stateDict['labelclassMap'] = classMap_updated
        return model, stateDict