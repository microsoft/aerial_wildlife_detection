#TODO: plug in RetinaNet implementation eventually...

#TODO 2: define function shells first in an abstract superclass (with documentation) and a template.

class RetinaNet:

    def __init__(self, properties):
        #TODO
        print('calling retinanet constructor')
        self.properties = properties
    

    def train(self, dbConnector, config, stateDict, data):
        #TODO
        print('I just received {} data items.'.format(len(data)))
        return 0


    def average_epochs(self):
        #TODO
        return 0

    
    def inference(self):
        #TODO
        return 0

    
    def rank(self):
        #TODO
        return 0