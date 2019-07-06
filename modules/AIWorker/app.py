'''
    2019 Benjamin Kellenberger
'''

import importlib
import json
from modules.AIWorker.backend.worker import functional
from modules.Database.app import Database


class AIWorker:

    def __init__(self, config, app):
        self.config = config
        self.dbConnector = Database(config)
        self._init_model_instance()
    

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
        
        # create AI model instance
        self.modelInstance = modelClass(modelOptions)



    def call_train(self, data):
        return functional._call_train(self.dbConnector, self.config, data, getattr(self.modelInstance, 'train'))
    


    def call_average_epochs(self, modelStates):
        return functional._call_average_epochs(self.dbConnector, self.config, modelStates, getattr(self.modelInstance, 'average_epochs'))



    def call_inference(self, imageIDs):
        return functional._call_inference(self.dbConnector, self.config, imageIDs, getattr(self.modelInstance, 'inference'))

    

    def call_rank(self, data):
        return functional._call_rank(self.dbConnector, self.config, data, getattr(self.modelInstance, 'rank'))