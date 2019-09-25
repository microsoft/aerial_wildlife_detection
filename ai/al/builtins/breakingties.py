'''
    Implementation of the Breaking Ties heuristic
    (Luo et al. 2005: "Active Learning to Recognize Multiple Types of Plankton." JMLR 6, 589-613.)

    2019 Benjamin Kellenberger
'''

from ai.al.functional.noarch.functional import _breaking_ties

class BreakingTies:
    
    def __init__(self, project, config, dbConnector, fileServer, options):
        pass

    
    def rank(self, data, updateStateFun, **kwargs):
        
        # iterate through the images and predictions
        for imgID in data.keys():
            if 'predictions' in data[imgID]:
                for p in range(len(data[imgID]['predictions'])):
                    btVal = _breaking_ties(data[imgID]['predictions'][p])
                    data[imgID]['predictions'][p]['priority'] = btVal
        return data