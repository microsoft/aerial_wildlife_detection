'''
    Abstract base class implementation of a model trainer that 
    uses a tensorflow model.

    2019 Colin Torney
'''

import io
from ai.models import AIModel
from util.helpers import get_class_executable, check_args
import torch


class GenericTFModel(AIModel):

    model_class = None

    def __init__(self, config, dbConnector, fileServer, options, defaultOptions):
        super(GenericTFModel, self).__init__(config, dbConnector, fileServer, options)

        # parse the options and compare with the provided defaults
        self.options = check_args(self.options, defaultOptions)

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
 #       if 'cuda' in device and not torch.cuda.is_available():
 #           device = 'cpu'
        return device

    
    def initializeModel(self, stateDict, data, width, height):
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
  #          stateDict = json.loads(stateDict.decode('utf-8'))
            model = self.model_class.loadFromStateDict(stateDict, width, height)
            
            # mapping labelclass (UUID) to index in model (number)
            labelclassMap = stateDict['labelclassMap']
        else:
            # create new label class map
            labelclassMap = {}
            for idx, lcID in enumerate(data['labelClasses']):
                labelclassMap[lcID] = idx       #NOTE: we do not use the labelclass' serial 'idx', since this might contain gaps
            self.options['model']['kwargs']['labelclassMap'] = labelclassMap

            # initialize a fresh model
            model = self.model_class.loadFromStateDict(self.options['model']['kwargs'], width, height)

        return model, labelclassMap

    
    def exportModelState(self, model):
        '''
            Retrieves a state dict from the model (e.g. after training) and converts it
            to a byte array that can be sent back to the AIWorker, and eventually the
            database.
            Also puts the model back on CPU and empties the CUDA cache (if available).
        '''

        bio = io.BytesIO()
        torch.save(model.getStateDict(), bio) # replace with something none torch
        
  #      print(model.getStateDict())
  #      inter = json.dumps(model.getStateDict())
  #      print('dumped')
  #      inter2 = inter.encode('utf-8')
  #      print('encoded')
        return bio.getvalue()
    
    def average_model_states(self, stateDicts):
        '''
            Receives a list of model states (as bytes) and calls the model's
            averaging function to return a single, unified model state.
        '''

        # read state dicts from bytes
        for s in range(len(stateDicts)):
            stateDict = io.BytesIO(stateDicts[s])
#            stateDicts[s] = torch.load(stateDict, map_location=lambda storage, loc: storage)

        average_states = self.model_class.averageStateDicts(stateDicts)

        # all done; return state dict as bytes
        bio = io.BytesIO()
 #       torch.save(average_states, bio)
        return bio.getvalue()
