'''
    Java-ish singleton implementation. See https://www.python.org/download/releases/2.2/descrintro/#__new__
    and https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
'''

class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it
    def init(self, *args, **kwds):
        pass