'''
    Helper snippets for built-in AL heuristics on computing the priority score.

    2019-21 Benjamin Kellenberger
'''

from collections.abc import Iterable
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
        # criterion across multiple inputs (e.g., segmentation mask): take average
        btVal = np.mean(btVal)
        # btVal = btVal.tolist()
    return btVal


def _max_confidence(prediction):
    '''
        Returns the maximum value of the logits as a priority value.
    '''
    if 'logits' in prediction:
        if isinstance(prediction['logits'], Iterable):
            maxVal = max(prediction['logits'])
        else:
            try:
                maxVal = float(prediction['logits'])
            except:
                maxVal = None
    elif 'confidence' in prediction:
        if isinstance(prediction['confidence'], Iterable):
            maxVal = max(prediction['confidence'])
        else:
            try:
                maxVal = float(prediction['confidence'])
            except:
                maxVal = None
    else:
        maxVal = None
    
    if isinstance(maxVal, np.ndarray):
        # criterion across multiple inputs (e.g., segmentation mask): take maximum
        maxVal = np.max(maxVal)
        # maxVal = maxVal.tolist()
    return maxVal