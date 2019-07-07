#TODO: plug in RetinaNet implementation eventually...

#TODO 2: define function shells first in an abstract superclass (with documentation) and a template.

from celery import current_task

class RetinaNet:

    def __init__(self, config, dbConnector, fileServer, options):
        #TODO
        print('calling retinanet constructor')
        self.config = config
        self.dbConnector = dbConnector
        self.fileServer = fileServer
        self.options = options
    

    def train(self, stateDict, data):
        #TODO
        print('I just received {} data items.'.format(len(data)))


        #TODO: how to load an image in the worker
        import io
        bytea = self.fileServer.getFile(data[0]['filename'])
        from PIL import Image
        img = Image.open(io.BytesIO(bytea))
        print(img.size)


        import time
        n = 30
        for i in range(0, n):
            #TODO: might be useful for task logging: https://www.distributedpython.com/2018/11/06/celery-task-logger-format/
            # version below somehow doesn't work
            current_task.update_state(state='PROGRESS', meta={'done': i, 'total': n})
            time.sleep(1)

        return 0


    def average_model_states(self, stateDicts):
        #TODO
        return 0

    
    def inference(self, stateDict, data):
        #TODO
        return 0

    
    def rank(self, data):
        #TODO
        return 0