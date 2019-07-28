'''
    Runs one or more of the tiers (LabelingUI, AIController, AIWorker), based on the arguments passed.

    2019 Benjamin Kellenberger
'''

import os
import argparse
from bottle import Bottle
from util.configDef import Config


class Launcher:

    def __init__(self, args):
        self.args = args

        # load configuration
        self.config = Config()

        self.instances = []
        self._launch_instances()


    def _verify_unique(self, moduleClass):
        '''
            Compares the newly requested module, address and port against
            already launched modules on this instance.
            Raises an Exception if another module from the same type has already been launched on this instance
        '''
        for i in self.instances:
            if moduleClass.__class__.__name__ == i.__class__.__name__:
                raise Exception('Module {} already launched on this server.'.format(moduleClass.__class__.__name__))


    def _launch_instances(self):

        # prepare bottle
        app = Bottle()

        # parse requested instances
        instance_args = self.args.instance.split(',')

        # create user handler
        userHandler = REGISTERED_MODULES['UserHandler'](self.config, app)

        for i in instance_args:

            moduleName = i.strip()
            if moduleName == 'UserHandler':
                continue

            moduleClass = REGISTERED_MODULES[moduleName]
            
            # verify
            self._verify_unique(moduleClass)

            # create instance
            instance = moduleClass(self.config, app)
            self.instances.append(instance)

            # add authentication functionality
            if hasattr(instance, 'addLoginCheckFun'):
                instance.addLoginCheckFun(userHandler.checkAuthenticated)

        # run server
        host = self.config.getProperty('Server', 'host')
        port = self.config.getProperty('Server', 'port')
        app.run(host=host, port=port)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CV4Wildlife AL Service.')
    parser.add_argument('--settings_filepath', type=str, default='settings_wcsaerialblobs.ini', const=1, nargs='?',
                    help='Directory of the settings.ini file used for this machine (default: "config/settings.ini").')
    parser.add_argument('--instance', type=str, default='LabelUI', const=1, nargs='?',
                    help='Instance type(s) to run on this host. Accepts multiple keywords, comma-separated (default: "LabelUI").')
    args = parser.parse_args()

    os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)


    #TODO: dirty hack; have to import modules here for missing environment variable in celery_interface.py
    from modules import REGISTERED_MODULES


    Launcher(args)