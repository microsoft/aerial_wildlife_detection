'''
    Interface to the config.ini file.

    2019-20 Benjamin Kellenberger
'''

import os
from configparser import ConfigParser


class Config():

    def __init__(self):
        if not 'AIDE_CONFIG_PATH' in os.environ:
            raise ValueError('Missing system environment variable "AIDE_CONFIG_PATH".')
        self.config = ConfigParser()
        self.config.read(os.environ['AIDE_CONFIG_PATH'])


    def getProperty(self, module, propertyName, type=str, fallback=None):
        m = module

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
    parser.add_argument('--settings_filepath', type=str,
                    help='Directory of the settings.ini file used for this machine (default: "config/settings.ini").')
    parser.add_argument('--section', type=str, help='Configuration file section')
    parser.add_argument('--parameter', type=str, help='Parameter within the section')
    parser.add_argument('--fallback', type=str, help='Fallback value, if parameter does not exist (optional)')
    args = parser.parse_args()

    if 'settings_filepath' in args and args.settings_filepath is not None:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    if args.section is None or args.parameter is None:
        print('Usage: python configDef.py --section=<.ini file section> --parameter=<section parameter name> [--fallback=<default value>]')

    else:
        print(Config().getProperty(args.section, args.parameter, fallback=args.fallback))