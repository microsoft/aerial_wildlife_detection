''' Functionals '''
def _unNormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor



''' Class definitions '''
class UnNormalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    
    def __call__(self, tensor):
        return _unNormalize(tensor, self.mean, self.std)