'''
    Helper snippets for built-in AL heuristics on computing the priority score.

    2019-21 Benjamin Kellenberger
'''

import numpy as np


def _breaking_ties(prediction):
    '''
        Computes the Breaking Ties heuristic
        (Luo et al. 2005: "Active Learning to Recognize Multiple Types of Plankton." JMLR 6, 589-613.)

        In case of segmentation masks, the average BT value is returned.
    '''
    btVal = None
    if 'logits' in prediction:
        logits = np.array(prediction['logits'].copy())

        if logits.ndim == 3:
            # spatial prediction
            logits = np.sort(logits, 0)
            btVal = 1 - np.mean(logits[-1,...] - logits[-2,...])
        else:
            logits = np.sort(logits)
            btVal = 1 - (logits[-1] - logits[-2])
    if isinstance(btVal, np.ndarray):
        btVal = btVal.tolist()
    return btVal


def _max_confidence(prediction):
    '''
        Returns the maximum value of the logits as a priority value.
    '''
    if 'logits' in prediction:
        maxVal = max(prediction['logits'])
    elif 'confidence' in prediction:
        maxVal = max(prediction['confidence'])
    else:
        maxVal = None
    
    if isinstance(maxVal, np.ndarray):
        maxVal = maxVal.tolist()
    return maxVal