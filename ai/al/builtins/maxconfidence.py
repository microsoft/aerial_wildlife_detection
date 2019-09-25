'''
    Simply returns the maximum confidence value as a 'priority' score.

    2019 Benjamin Kellenberger
'''

from ai.al.functional.noarch.functional import _max_confidence

class MaxConfidence:

    def __init__(self, project, config, dbConnector, fileServer, options):
        pass

    
    def rank(self, data, updateStateFun, **kwargs):
        
        # iterate through the images and predictions
        for imgID in data.keys():
            if 'predictions' in data[imgID]:
                for p in range(len(data[imgID]['predictions'])):
                    btVal = _max_confidence(data[imgID]['predictions'][p])
                    data[imgID]['predictions'][p]['priority'] = btVal
        return data