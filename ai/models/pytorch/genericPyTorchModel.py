'''
    Abstract base class implementation of a model trainer that 
    uses a PyTorch model.
    This class merely provides some helpers and defaults that can be (re-) used
    by subclassing PyTorch implementations, such as option checks, model initiali-
    zations, model state exports, etc.

    2019-20 Benjamin Kellenberger
'''

import io
import torch
from torch.optim import SGD
from ai.models import AIModel
from ai.models.pytorch import parse_transforms
from util.helpers import get_class_executable, check_args
from util import optionsHelper



class GenericPyTorchModel(AIModel):

    '''
        New version of the base class for the (built-in) PyTorch models that
        implements the GUI-enhanced model options.
    '''

    model_class = None

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(GenericPyTorchModel, self).__init__(project, config, dbConnector, fileServer, options)

        # try to fill and substitute global definitions in JSON-enhanced options
        if isinstance(options, dict) and 'defs' in options:
            try:
                updatedOptions = optionsHelper.substitute_definitions(options.copy())
                self.options = updatedOptions
            except:
                # something went wrong; ignore
                pass

        # retrieve executables
        try:
            self.model_class = get_class_executable(optionsHelper.get_hierarchical_value(self.options, ['options', 'model', 'class']))
        except:
            self.model_class = None
        try:
            self.criterion_class = get_class_executable(optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'criterion', 'class']))
        except:
            self.criterion_class = None
        try:
            self.optim_class = get_class_executable(optionsHelper.get_hierarchical_value(self.options, ['options', 'train', 'optim']))
        except:
            self.optim_class = SGD
        try:
            self.dataset_class = get_class_executable(optionsHelper.get_hierarchical_value(self.options, ['options', 'dataset']))
        except:
            self.dataset_class = None


    def get_device(self):
        device = optionsHelper.get_hierarchical_value(self.options, ['options', 'general', 'device', 'value', 'id'])
        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'
        return device

    
    def initializeModel(self, stateDict, data, addMissingLabelClasses=False, removeObsoleteLabelClasses=False):
        '''
            Converts the provided stateDict from a bytes array to a torch-loadable
            object and initializes a model from it. Also returns a 'labelClassMap',
            defining the indexing between label classes and the model.
            If the stateDict object is None, a new model and labelClassMap are crea-
            ted from the defaults.

            If "addMissingLabelClasses" is True, new output neurons are appended for
            label classes that are not present in the model's current labelclass map,
            and the map is updated.
            Likewise, if "removeObsoleteLabelClasses" is True, existing outputs for
            label classes that are not present in the set of label classes anymore are
            removed, and the map is also updated.
        '''
        # initialize model
        if stateDict is not None:
            stateDict = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
            model = self.model_class.loadFromStateDict(stateDict)
            
            # mapping labelclass (UUID) to index in model (number)
            labelclassMap = stateDict['labelclassMap']

            if addMissingLabelClasses or removeObsoleteLabelClasses:
                # modification of model outputs
                if hasattr(model, 'updateModel'):
                    model.updateModel(data['labelClasses'], addMissingLabelClasses, removeObsoleteLabelClasses)

        else:
            # create new label class map
            labelclassMap = {}
            for idx, lcID in enumerate(data['labelClasses']):
                labelclassMap[lcID] = idx       #NOTE: we do not use the labelclass' serial 'idx', since this might contain gaps
            # self.options['options']['model']['labelclassMap'] = labelclassMap   #TODO: never used and obsolete with new options format

            # initialize a fresh model
            modelKwargs = {}
            modelOptions = optionsHelper.get_hierarchical_value(self.options, ['options', 'model'], [])
            for key in modelOptions.keys():
                if key not in optionsHelper.RESERVED_KEYWORDS:
                    modelKwargs[key] = optionsHelper.get_hierarchical_value(modelOptions[key], ['value', 'id'])
            modelKwargs['labelclassMap'] = labelclassMap

            model = self.model_class(**modelKwargs)

        return model, labelclassMap


    @staticmethod
    def parseTransforms(transforms):
        '''
            Retrieves a list of transform definitions according to the new GUI-enhanced
            JSON options and tries to parse them.
            Supports torchvision and built-in transforms. Takes special care of inputs
            for certain transformations that require iterable inputs, such as the three
            "mean" and "std" values of "torchvision.transforms.Normalize".
            Returns a list with instances of all transforms parsed in order.
        '''
        def _parse_transform(tr):
            tr_kwargs = {}
            if isinstance(tr, dict):
                tr_class = optionsHelper.get_hierarchical_value(tr, ['id'])
                for key in tr.keys():
                    if key not in optionsHelper.RESERVED_KEYWORDS:
                        tr_kwargs[key] = optionsHelper.get_hierarchical_value(tr[key], ['value'])
            elif isinstance(tr, str):
                tr_class = tr
            if tr_class.endswith('DefaultTransform'):
                subTr = GenericPyTorchModel.parseTransforms(tr['transform'])
                tr_kwargs = {'transform': subTr}    
            elif tr_class == 'torchvision.transforms.Normalize':
                mean = optionsHelper.get_hierarchical_value(tr_kwargs, ['mean', 'value'])
                std = optionsHelper.get_hierarchical_value(tr_kwargs, ['std', 'value'])
                tr_kwargs = {
                    'mean': [
                        mean[0]['value'],
                        mean[1]['value'],
                        mean[2]['value']
                    ],
                    'std': [
                        std[0]['value'],
                        std[1]['value'],
                        std[2]['value']
                    ]
                }
            elif tr_class == 'torchvision.transforms.ColorJitter':
                brightness = optionsHelper.get_hierarchical_value(tr_kwargs, ['brightness', 'value'])
                contrast = optionsHelper.get_hierarchical_value(tr_kwargs, ['contrast', 'value'])
                saturation = optionsHelper.get_hierarchical_value(tr_kwargs, ['saturation', 'value'])
                hue = optionsHelper.get_hierarchical_value(tr_kwargs, ['hue', 'value'])
                tr_kwargs = {
                    'brightness': [
                        brightness[0]['value'],
                        brightness[1]['value'],
                    ],
                    'contrast': [
                        contrast[0]['value'],
                        contrast[1]['value'],
                    ],
                    'saturation': [
                        saturation[0]['value'],
                        saturation[1]['value'],
                    ],
                    'hue': [
                        hue[0]['value'],
                        hue[1]['value'],
                    ]
                }
            elif tr_class.endswith('Compose'):
                sub_transforms = GenericPyTorchModel.parseTransforms(tr['transforms'])
                tr_kwargs = {'transforms': sub_transforms}
            #TODO: others?

            return parse_transforms({
                'class': tr_class,
                'kwargs': tr_kwargs
            })

        if isinstance(transforms, dict) or isinstance(transforms, str):
            return _parse_transform(transforms)
        else:
            transforms_out = []
            for tr in transforms:
                trInst = _parse_transform(tr)
                transforms_out.append(trInst)
            return transforms_out

    
    def exportModelState(self, model):
        '''
            Retrieves a state dict from the model (e.g. after training) and converts it
            to a byte array that can be sent back to the AIWorker, and eventually the
            database.
            Also puts the model back on CPU and empties the CUDA cache (if available).
        '''
        if 'cuda' in self.get_device():
            torch.cuda.empty_cache()
        model.cpu()

        bio = io.BytesIO()
        torch.save(model.getStateDict(), bio)

        return bio.getvalue()

    
    def average_model_states(self, stateDicts, updateStateFun):
        '''
            Receives a list of model states (as bytes) and calls the model's
            averaging function to return a single, unified model state.
        '''

        # read state dicts from bytes
        for s in range(len(stateDicts)):
            stateDict = io.BytesIO(stateDicts[s])
            stateDicts[s] = torch.load(stateDict, map_location=lambda storage, loc: storage)

        average_states = self.model_class.averageStateDicts(stateDicts)

        # all done; return state dict as bytes
        bio = io.BytesIO()
        torch.save(average_states, bio)
        return bio.getvalue()



class GenericPyTorchModel_Legacy(AIModel):

    '''
        NOTE: This is a legacy base class that works with the old, non-GUI-enhanced
        JSON settings options. As soon as all the built-in models are converted to
        the new style, this class will be removed.
        Please see "genericPyTorchModel.py" for replacement.
    '''

    model_class = None

    def __init__(self, project, config, dbConnector, fileServer, options, defaultOptions=None):
        super(GenericPyTorchModel_Legacy, self).__init__(project, config, dbConnector, fileServer, options)

        # parse the options and compare with the provided defaults (if provided)
        if defaultOptions is not None:
            self.options = check_args(self.options, defaultOptions)
        else:
            self.options = options

        # retrieve executables
        try:
            self.model_class = get_class_executable(self.options['model']['class'])
        except:
            self.model_class = None
        try:
            self.criterion_class = get_class_executable(self.options['train']['criterion']['class'])
        except:
            self.criterion_class = None
        try:
            self.optim_class = get_class_executable(self.options['train']['optim']['class'])
        except:
            self.optim_class = SGD
        try:
            self.dataset_class = get_class_executable(self.options['dataset']['class'])
        except:
            self.dataset_class = None


    def get_device(self):
        device = self.options['general']['device']
        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'
        return device

    
    def initializeModel(self, stateDict, data):
        '''
            Converts the provided stateDict from a bytes array to a torch-loadable
            object and initializes a model from it. Also returns a 'labelClassMap',
            defining the indexing between label classes and the model.
            If the stateDict object is None, a new model and labelClassMap are crea-
            ted from the defaults.
        '''

        # initialize model
        if stateDict is not None:
            stateDict = torch.load(io.BytesIO(stateDict), map_location=lambda storage, loc: storage)
            model = self.model_class.loadFromStateDict(stateDict)
            
            # mapping labelclass (UUID) to index in model (number)
            labelclassMap = stateDict['labelclassMap']
        else:
            # create new label class map
            labelclassMap = {}
            for idx, lcID in enumerate(data['labelClasses']):
                labelclassMap[lcID] = idx       #NOTE: we do not use the labelclass' serial 'idx', since this might contain gaps
            # self.options['model']['kwargs']['labelclassMap'] = labelclassMap

            # initialize a fresh model
            model = self.model_class.loadFromStateDict(self.options['model']['kwargs'])

        return model, labelclassMap

    
    def exportModelState(self, model):
        '''
            Retrieves a state dict from the model (e.g. after training) and converts it
            to a byte array that can be sent back to the AIWorker, and eventually the
            database.
            Also puts the model back on CPU and empties the CUDA cache (if available).
        '''
        if 'cuda' in self.get_device():
            torch.cuda.empty_cache()
        model.cpu()

        bio = io.BytesIO()
        torch.save(model.getStateDict(), bio)

        return bio.getvalue()

    
    def average_model_states(self, stateDicts, updateStateFun):
        '''
            Receives a list of model states (as bytes) and calls the model's
            averaging function to return a single, unified model state.
        '''

        # read state dicts from bytes
        for s in range(len(stateDicts)):
            stateDict = io.BytesIO(stateDicts[s])
            stateDicts[s] = torch.load(stateDict, map_location=lambda storage, loc: storage)

        average_states = self.model_class.averageStateDicts(stateDicts)

        # all done; return state dict as bytes
        bio = io.BytesIO()
        torch.save(average_states, bio)
        return bio.getvalue()