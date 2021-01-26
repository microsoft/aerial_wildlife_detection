'''
    Interface to the config.ini file.

    2019-21 Benjamin Kellenberger
'''

import os
from configparser import ConfigParser
from util.helpers import LogDecorator


class Config():

    def __init__(self, override_config_path=None, verbose_start=False):
        if verbose_start:
            print('Reading configuration...'.ljust(6), end='')
        if isinstance(override_config_path, str) and len(override_config_path):
            configPath = override_config_path
        elif 'AIDE_CONFIG_PATH' in os.environ:
            configPath = os.environ['AIDE_CONFIG_PATH']
        else:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise ValueError('Neither system environment variable "AIDE_CONFIG_PATH" nor override path are set.')
        
        self.config = None
        try:
            self.config = ConfigParser()
            self.config.read(configPath)
        except Exception as e:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not read configuration file (message: "{str(e)}").')
        
        if verbose_start:
            LogDecorator.print_status('ok')


    def getProperty(self, module, propertyName, type=str, fallback=None):
        try:
            if type==bool:
                value = self.config.getboolean(module, propertyName, fallback=fallback)
            elif type==float:
                value = self.config.getfloat(module, propertyName, fallback=fallback)
            elif type==int:
                value = self.config.getint(module, propertyName, fallback=fallback)
            else:
                value = self.config.get(module, propertyName, fallback=fallback)
            if type is not None and not isinstance(value, type):
                return fallback
            else:
                return value
        except:
            return fallback



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
    parser.add_argument('--type', type=str, help='Parameter type. One of {"str" (default), "bool", "int", "float", None (everything else)}')
    parser.add_argument('--fallback', type=str, help='Fallback value, if parameter does not exist (optional)')
    args = parser.parse_args()

    if 'settings_filepath' in args and args.settings_filepath is not None:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    if args.section is None or args.parameter is None:
        print('Usage: python configDef.py --section=<.ini file section> --parameter=<section parameter name> [--fallback=<default value>]')

    else:
        try:
            type = args.type.lower()
            if type=='str':
                type=str
            elif type=='bool':
                type=bool
            elif type=='int':
                type=int
            elif type=='float':
                type=float
            else:
                type=None
        except:
            type=None

        print(Config().getProperty(args.section, args.parameter, type=type, fallback=args.fallback))