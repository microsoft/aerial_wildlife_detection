'''
    Helper snippets for built-in AL heuristics on computing the priority score.

    2019 Benjamin Kellenberger
'''

def _breaking_ties(prediction):
    '''
        Computes the Breaking Ties heuristic
        (Luo et al. 2005: "Active Learning to Recognize Multiple Types of Plankton." JMLR 6, 589-613.)
    '''
    btVal = None
    if 'logits' in prediction:
        logits = prediction['logits'].copy()
        logits.sort()
        btVal = 1 - (logits[-1] - logits[-2])
    return btVal


def _max_confidence(prediction):
    '''
        Returns the maximum value of the logits as a priority value.
    '''
    if 'logits' in prediction:
        return max(prediction['logits'])
    return None