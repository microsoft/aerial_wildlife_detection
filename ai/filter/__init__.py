'''
    Filters are classes that accept a dict of annotations made by (potentially multiple) users
    and reduce them to a common set, e.g. by taking only the most frequently assigned label to
    an image (classification), by returning the MBR of overlapping bounding boxes (detection), etc.

    2019 Benjamin Kellenberger
'''

class AbstractFilter:

    def __init__(self, config, dbConnector, fileServer, options):
        self.config = config
        self.dbConnector = dbConnector
        self.fileServer = fileServer
        self.options = options
    

    def filter(self, data, **kwargs):
        raise NotImplementedError('Not implemented for abstract base class.')