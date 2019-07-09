'''
    Composes multiple AL criteria and selects the maximum value over all.

    2019 Benjamin Kellenberger
'''

from util.helpers import get_class_executable

class Compose:

    def __init__(self, config, dbConnector, fileServer, options):
        
        # parse provided functions
        self.heuristics = []
        for h in options['rank']['heuristics']:
            self.heuristics.append(get_class_executable(h))

    
    def rank(self, data, **kwargs):
        
        # iterate through the images and predictions
        for imgID in data.keys():
            if 'predictions' in data[imgID]:
                for predID in data[imgID]['predictions'].keys():
                    pred = data[imgID]['predictions'][predID]

                    # iterate over heuristics and take the max
                    val = -1
                    for h in self.heuristics:
                        val = max(val, h(pred))
                    data[imgID]['predictions'][predID]['priority'] = val
        return data