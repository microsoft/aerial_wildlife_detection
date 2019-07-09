'''
    Simply returns the maximum confidence value as a 'priority' score.

    2019 Benjamin Kellenberger
'''

from ai.al.functional.noarch.functional import _max_confidence

class MaxConfidence:

    def __init__(self, config, dbConnector, fileServer, options):
        pass

    
    def rank(self, data, **kwargs):
        
        # iterate through the images and predictions
        for imgID in data.keys():
            if 'predictions' in data[imgID]:
                for predID in data[imgID]['predictions'].keys():
                    pred = data[imgID]['predictions'][predID]
                    val = _max_confidence(pred)
                    data[imgID]['predictions'][predID]['priority'] = val
        return data