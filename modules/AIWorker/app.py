'''
    2019 Benjamin Kellenberger
'''

import importlib
import inspect
import json
from modules.AIWorker.backend.worker import functional, fileserver
from modules.Database.app import Database


class AIWorker():

    def __init__(self, config, app):
        self.config = config
        self.dbConnector = Database(config)
        self._init_fileserver()
        self._init_model_instance()
    


    def _init_fileserver(self):
        '''
            The AIWorker has a special routine to detect whether the instance it is running on
            also hosts the file server. If it does, data are loaded directly from disk to avoid
            going through the loopback network.
        '''
        self.fileServer = fileserver.FileServer(self.config)


    def _init_model_instance(self):
        # parse AI model path
        try:
            modelOptions = json.load(self.config.getProperty('AIController', 'model_options_path'))
        except:
            modelOptions = None
        modelLibPath = self.config.getProperty('AIController', 'model_lib_path')

        # import superclass first, then retrieve class object
        superclass = importlib.import_module(modelLibPath[0:modelLibPath.rfind('.')])
        modelClass = getattr(superclass, modelLibPath[modelLibPath.rfind('.')+1:])

        # verify functions and arguments
        requiredFunctions = {
            '__init__' : ['config', 'dbConnector', 'fileServer', 'options'],
            'train' : ['stateDict', 'data'],
            'average_model_states' : ['stateDicts'],
            'inference' : ['stateDict', 'data'],
            'rank' : ['data']
        }   #TODO: make more elegant?
        functionNames = [func for func in dir(modelClass) if callable(getattr(modelClass, func))]

        for key in requiredFunctions:
            if not key in functionNames:
                raise Exception('Class {} is missing required function {}.'.format(modelLibPath, key))

            # check function arguments bidirectionally
            funArgs = inspect.getargspec(getattr(modelClass, key))
            for arg in requiredFunctions[key]:
                if not arg in funArgs.args:
                    raise Exception('Method {} of class {} is missing required argument {}.'.format(modelLibPath, key, arg))
            for arg in funArgs.args:
                if arg != 'self' and not arg in requiredFunctions[key]:
                    raise Exception('Unsupported argument {} of method {} in class {}.'.format(arg, key, modelLibPath))

        # create AI model instance
        self.modelInstance = modelClass(config=self.config, dbConnector=self.dbConnector, fileServer=self.fileServer, options=modelOptions)



    def call_train(self, data):
        return functional._call_train(self.dbConnector, self.config, data, getattr(self.modelInstance, 'train'), self.fileServer)
    


    def call_average_model_states(self, modelStates):
        return functional._call_average_model_states(self.dbConnector, self.config, modelStates, getattr(self.modelInstance, 'average_model_states'), self.fileServer)



    def call_inference(self, imageIDs):
        return functional._call_inference(self.dbConnector, self.config, imageIDs, getattr(self.modelInstance, 'inference'), self.fileServer)

    

    def call_rank(self, data):
        return functional._call_rank(self.dbConnector, self.config, data, getattr(self.modelInstance, 'rank'), self.fileServer)