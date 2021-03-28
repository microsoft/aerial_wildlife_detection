import numpy as np

def _breaking_ties(prediction):
    btVal = None
    if 'logits' in prediction:
        logits = np.array(prediction['logits'].copy())

        if logits.ndim == 3:
            logits = np.sort(logits, 0)
            btVal = 1 - np.mean(logits[-1,...] - logits[-2,...])
        else:
            logits = np.sort(logits)
            btVal = 1 - (logits[-1] - logits[-2])
    return btVal


def _max_confidence(prediction):
    if 'logits' in prediction:
        return max(prediction['logits'])
    return None
