'''
    Interface to the config.ini file.

    2019 Benjamin Kellenberger
'''

from configparser import ConfigParser
from modules import REGISTERED_MODULES


class Config():

    CONFIG_PATH = 'config/settings.ini'

    def __init__(self):
        self.config = ConfigParser()
        self.config.read(self.CONFIG_PATH)


    def getProperty(self, module, propertyName, type=str):

        if isinstance(module, str):
            m = module
        else:
            if hasattr(module, '__name__') and module.__name__ in REGISTERED_MODULES:
                m = module.__name__
            elif hasattr(module, '__class__') and module.__class__.__name__ in REGISTERED_MODULES:
                m = module.__class__.__name__
            else:
                m = module
            
            if not m in REGISTERED_MODULES:
                raise Exception('Module {} has not been registered.'.format(m))

        if type==bool:
            return self.config.getboolean(m, propertyName)
        elif type==float:
            return self.config.getfloat(m, propertyName)
        elif type==int:
            return self.config.getint(m, propertyName)
        else:
            return self.config.get(m, propertyName)