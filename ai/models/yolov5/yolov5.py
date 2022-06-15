'''
    Base model implementation of the YoloV5 library:
    https://github.com/ultralytics/yolov5

    2022 Benjamin Kellenberger
'''

import io
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# from yolov5.models.yolo import Model      #TODO: doesn't want to import

from ai.models import AIModel
from util import optionsHelper

from ai.models.yolov5._default_options import DEFAULT_OPTIONS
from ai.models.yolov5.dataset import getYOLOv5Data


class YOLOv5(AIModel):

    # configuration in YOLOv5 format
    DEFAULT_CONFIG = {
        'ch': 3,    # input channels
        'nc': 0,    #
        # 'anchors': None
        # 'depth_multiple': None,
        # 'width_multiple': None,
        # 'backbone': None,
        # 'head': None,
        'inplace': True
    }

    OPTIM_KWARGS = (
        'lr',
        'lr_decay',
        'weight_decay',
        'momentum',
        'dampening',
        'nesterov',
        'rho'
    )

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(YOLOv5, self).__init__(project, config, dbConnector, fileServer, options)
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except:
                # something went wrong; discard options #TODO
                options = None
        
        # try to fill and substitute global definitions in JSON-enhanced options
        if isinstance(options, dict):
            try:
                updatedOptions = optionsHelper.merge_options(self.getDefaultOptions(), options.copy())
                updatedOptions = optionsHelper.substitute_definitions(updatedOptions)
                self.options = updatedOptions
            except:
                # something went wrong; ignore
                pass

        # verify options
        result = self.verifyOptions(self.options)
        if not result['valid']:
            print(f'[{self.project}] WARNING: provided options are invalid; replacing with defaults...')
            self.options = self.getDefaultOptions()
    
        # extract YOLOv5 cfg subset
        self.yolov5cfg = self._extract_hyp_options(self.options['options']['train'])
        if 'anchors' not in self.yolov5cfg:
            self.yolov5cfg['anchors'] = 3
        
        #TODO: evolve: https://github.com/ultralytics/yolov5/blob/master/train.py#L587


    @classmethod
    def _extract_hyp_options(cls, options, hyp={}):
        for key in options.keys():
            if key.startswith('yolov5.hyp'):
                hyp[key.replace('yolov5.hyp.', '')] = optionsHelper.get_hierarchical_value(options[key], ['value'])
            elif isinstance(options[key], dict):
                cls._extract_hyp_options(options[key], hyp)
        return hyp


    @classmethod
    def getDefaultOptions(cls):
        #TODO: json file
        return optionsHelper.substitute_definitions(DEFAULT_OPTIONS)
        # try:
        #     # try to load defaults from JSON file first
        #     options = json.load(open(jsonFilePath, 'r'))
        # except Exception as e:
        #     # error; fall back to built-in defaults
        #     print(f'Error reading default options file "{jsonFilePath}" (message: "{str(e)}"), falling back to built-in options.')
        #     options = DEFAULT_OPTIONS
        
        # # expand options
        # options = optionsHelper.substitute_definitions(options)

        # return options
    

    @classmethod
    def verifyOptions(cls, options):
        defaultOptions = cls.getDefaultOptions()
        if options is None:
            return {
                'valid': True,
                'options': defaultOptions
            }
        try:
            if isinstance(options, str):
                options = json.loads(options)
            options = optionsHelper.substitute_definitions(options)
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Options are not in a proper format (message: {str(e)}).']
            }
        try:
            # mandatory field: model config
            modelConfig = optionsHelper.get_hierarchical_value(options, ['options', 'model', 'config', 'value'])
            if modelConfig is None:
                raise Exception('missing model type field in options.')

            opts, warnings, errors = optionsHelper.verify_options(options['options'], autoCorrect=True)
            options['options'] = opts
            return {
                'valid': not len(errors),
                'warnings': warnings,
                'errors': errors,
                'options': options
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'An error occurred trying to verify options (message: {str(e)}).']
            }
    

    def initializeModel(self, stateDict, data, revertProjectToStateMap=False):
        '''
            Loads Bytes object "stateDict" through torch and looks for a YOLOv5
            config to initialize the model structure, optionally with pre-trained
            weights if available.
            Returns the model instance, the state dict, inter alia augmented with a
            'labelClassMap', defining the indexing between label classes and the model,
            and a list of new label classes (i.e., those that are present in the "data"
            dict, but not in the model definition).
            If the stateDict object is None, a new model and labelClassMap are crea-
            ted from the defaults.

            Note that this function does NOT modify the model w.r.t. the actually
            present label classes in the project. This functionality requires re-
            configuring the model's prediction output and depends on the model imple-
            mentation.
        '''
        #TODO: force new model
        forceNewModel = False

        # load state dict
        if stateDict is not None and not forceNewModel:
            if not isinstance(stateDict, dict):
                stateDict = torch.load(io.BytesIO(stateDict), map_location='cpu')
        else:
            stateDict = {}

        # retrieve YOLOv5 cfg
        if 'yolov5cfg' in stateDict and not forceNewModel:
            # medium priority: model overrides
            self.yolov5cfg.update(stateDict['yolov5cfg'])

        # top priority: AIDE config overrides
        #TODO

        stateDict['yolov5cfg'] = self.yolov5cfg

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        #TODO

        # input channels
        ch = len(data.get('band_config', ['r','g','b']))        #TODO: make more advanced

        # number of classes
        nc = len(data['labelClasses'])                          #TODO: ditto

        self.yolov5cfg.update({
            'nc': nc,
            'ch': ch
        })

        # construct model and load state
        modelType = optionsHelper.get_hierarchical_value(self.options['options'], ['model', 'config', 'id'], fallback='yolov5s')
        model = torch.hub.load('ultralytics/yolov5', modelType)

        # model = Model(self.yolov5cfg,
        #                 ch=self.yolov5cfg.get('ch', 3),
        #                 nc=self.yolov5cfg.get('nc', 1),                 #TODO
        #                 anchors=self.yolov5cfg.get('anchors', 3))       #TODO

        if 'model' in stateDict and not forceNewModel:
            model.load_state_dict(stateDict['model'])
        
        #TODO: continue: https://github.com/ultralytics/yolov5/blob/master/train.py#L124
        # from utils.autobatch import check_train_batch_size


        # load or create labelclass map
        if 'labelclassMap' in stateDict and not forceNewModel:
            labelclassMap = stateDict['labelclassMap']
        else:
            labelclassMap = {}

            # add existing label classes; any non-UUID entry will be discarded during prediction
            try:
                pretrainedClasses = model.names
                for idx, cID in enumerate(pretrainedClasses):
                    labelclassMap[cID] = idx
            except:
                pass

        # check data for new label classes
        projectToStateMap = {}
        newClasses = []
        for lcID in data['labelClasses']:
            if lcID not in labelclassMap:
                # check if original label class got re-mapped
                lcMap_origin_id = data['labelClasses'][lcID].get('labelclass_id_model', None)
                if lcMap_origin_id is None or lcMap_origin_id not in labelclassMap:
                    # no remapping done; class really is new
                    newClasses.append(lcID)
                else:
                    # class has been re-mapped
                    if revertProjectToStateMap:
                        projectToStateMap[lcMap_origin_id] = lcID
                    else:
                        projectToStateMap[lcID] = lcMap_origin_id
        stateDict['labelclassMap'] = labelclassMap

        return model, stateDict, newClasses, projectToStateMap

        

    def train(self, stateDict, data, updateStateFun):
        '''
            Main training function.
        '''
        # initialize model
        model, stateDict, _, projectToStateMap = self.initializeModel(stateDict, data, False)
        
        # wrap dataset for usage with YOLOv5
        ignoreUnsure = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'ignore_unsure', 'value'], fallback=True)
        filterEmpty = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'filter_empty', 'value'], fallback=False)
        batchSize = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'batch_size', 'value'], fallback=2)     #TODO: use YOLOv5's auto batch size
        dataset = getYOLOv5Data(data, stateDict['labelclassMap'], projectToStateMap, ignoreUnsure, filterEmpty)
        dataLoader = DataLoader(
            dataset,
            batch_size=batchSize,
            shuffle=True
        )
        numImg = len(data['images'])

        # optim     #TODO: check https://github.com/ultralytics/yolov5/blob/master/train.py#L156
        optimClass = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'optim', 'class'], fallback='SGD')
        configKwargs = optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'optim', 'value'])
        optimKwargs = {}
        for key in self.OPTIM_KWARGS:
            if key in configKwargs:
                val = optionsHelper.get_hierarchical_value(configKwargs[key], ['value'], fallback=None)
                if val is not None:
                    optimKwargs[key] = val
        optim = getattr(torch.optim, optimClass)(model.parameters(), **optimKwargs)

        #TODO: scheduler: https://github.com/ultralytics/yolov5/blob/master/train.py#L178

        # train
        model.train()
        for idx, (images, target) in enumerate(tqdm(dataLoader)):
            
            images, target = images.to(self.device), target.to(self.device)

            mloss = torch.zeros(3, device=self.device)  # mean losses




    def inference(self, stateDict, data, updateStateFun):
        #TODO
        pass