'''
    Main Bottle and routings for the AIController instance.

    2019 Benjamin Kellenberger
'''

from bottle import post, request, response
from modules.AIController.backend.middleware import AIMiddleware


class AIController:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self.middleware = AIMiddleware(config)

        self._initBottle()


    def _initBottle(self):
        
        @self.app.get('/startTraining')     #TODO: POST
        def start_training():
            '''
                Manually request AIController to train the model.
                This still only works if there is no training process ongoing.
                Otherwise the request is aborted.
            '''
            #TODO: logincheck
            # try:
            status = self.middleware.start_training(minTimestamp='lastState', distributeTraining=False) #TODO
            # except Exception as e:
            #     status = str(e)
            return { 'status' : status }

        
        @self.app.get('/startInference')
        def start_inference():
            '''
                TODO: just here for debugging purposes; in reality inference should automatically be called after training
            '''
            status = self.middleware.start_inference(forceUnlabeled=True, maxNumImages=None, maxNumWorkers=1)
            return { 'status' : status }


        @self.app.get('/checkStatus')    #TODO: POST
        def check_status():
            '''
                Queries the middleware for any ongoing training worker processes
                and returns the stati of each in a dict.
            '''
            #TODO: logincheck
            # try:
            status = self.middleware.check_status(True, True)   #TODO: args
            # except Exception as e:
            #     status = str(e)
            
            return { 'status' : status }