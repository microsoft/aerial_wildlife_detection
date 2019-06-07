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


    def getProperty(self, module, propertyName, type=str, fallback=None):
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
            return self.config.getboolean(m, propertyName, fallback=fallback)
        elif type==float:
            return self.config.getfloat(m, propertyName, fallback=fallback)
        elif type==int:
            return self.config.getint(m, propertyName, fallback=fallback)
        else:
            return self.config.get(m, propertyName, fallback=fallback)



if __name__ == '__main__':
    '''
        Read default config and return parameters as specified through arguments.
        This will be used to e.g. set up the SQL database.
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Get configuration entry programmatically.')
    parser.add_argument('--section', type=str, help='Configuration file section')
    parser.add_argument('--parameter', type=str, help='Parameter within the section')
    args = parser.parse_args()

    if args.section is None or args.parameter is None:
        print('Usage: python configDef.py --section=<.ini file section> --parameter=<section parameter name>')

    else:
        print(Config().getProperty(args.section, args.parameter))