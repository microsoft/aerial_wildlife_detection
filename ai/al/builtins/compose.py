'''
    Composes multiple AL criteria and selects the maximum value over all.

    2019 Benjamin Kellenberger
'''

from util.helpers import get_class_executable

class Compose:

    def __init__(self, project, config, dbConnector, fileServer, options):
        
        # parse provided functions
        self.heuristics = []
        for h in options['rank']['heuristics']:
            self.heuristics.append(get_class_executable(h))

    
    def rank(self, data, updateStateFun, **kwargs):
        
        # iterate through the images and predictions
        for imgID in data.keys():
            if 'predictions' in data[imgID]:
                for p in range(len(data[imgID]['predictions'])):
                    # iterate over heuristics and take the max
                    val = -1
                    for h in self.heuristics:
                        val = max(val, h(data[imgID]['predictions'][p]))
                    data[imgID]['predictions'][p]['priority'] = val
        return data