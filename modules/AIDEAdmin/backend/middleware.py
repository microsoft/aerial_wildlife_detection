'''
    Middleware for administrative functionalities
    of AIDE.

    2020 Benjamin Kellenberger
'''

import os
import requests
from constants.version import AIDE_VERSION
from modules.Database.app import Database
from util.helpers import is_localhost


class AdminMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    def getServiceDetails(self, warn_version_mismatch=False):
        '''
            Queries the indicated AIController and FileServer
            modules for availability and their version. Returns
            metadata about the setup of AIDE accordingly.
            Raises an Exception if not running on the main host.
            If "warn_version_mismatch" is True, a warning state-
            ment is printed to the command line if the version of
            AIDE on the attached AIController and/or FileServer
            is not the same as on the host.
        '''
        # check if running on the main host
        modules = os.environ['AIDE_MODULES'].strip().split(',')
        modules = set([m.strip() for m in modules])
        if not 'LabelUI' in modules:
            # not running on main host
            raise Exception('Not a main host; cannot query service details.')

        aic_uri = self.config.getProperty('Server', 'aiController_uri', type=str, fallback=None)
        fs_uri = self.config.getProperty('Server', 'dataServer_uri', type=str, fallback=None)

        if not is_localhost(aic_uri):
            # AIController runs on a different machine; poll for version of AIDE
            try:
                aic_response = requests.get(os.path.join(aic_uri, 'version'))
                aic_version = aic_response.text
                if warn_version_mismatch and aic_version != AIDE_VERSION:
                    print('WARNING: AIDE version of connected AIController differs from main host.')
                    print(f'\tAIController URI: {aic_uri}')
                    print(f'\tAIController AIDE version:    {aic_version}')
                    print(f'\tAIDE version on this machine: {AIDE_VERSION}')
            except Exception as e:
                print(f'WARNING: error connecting to AIController (message: "{str(e)}").')
                aic_version = None
        else:
            aic_version = AIDE_VERSION
        if not is_localhost(fs_uri):
            # same for the file server
            try:
                fs_response = requests.get(os.path.join(fs_uri, 'version'))
                fs_version = fs_response.text
                if warn_version_mismatch and aic_version != AIDE_VERSION:
                    print('WARNING: AIDE version of connected FileServer differs from main host.')
                    print(f'\tFileServer URI: {fs_uri}')
                    print(f'\tFileServer AIDE version:       {fs_version}')
                    print(f'\tAIDE version on this machine: {AIDE_VERSION}')
            except Exception as e:
                print(f'WARNING: error connecting to FileServer (message: "{str(e)}").')
                fs_version = None
        else:
            fs_version = AIDE_VERSION

        # query database
        dbVersion = self.dbConnector.execute('SHOW server_version;', None, 1)[0]['server_version']
        try:
            dbVersion = dbVersion.split(' ')[0].strip()
        except:
            pass
        dbInfo = self.dbConnector.execute('SELECT version() AS version;', None, 1)[0]['version']

        return {
                'aide_version': AIDE_VERSION,
                'AIController': {
                    'uri': aic_uri,
                    'aide_version': aic_version
                },
                'FileServer': {
                    'uri': fs_uri,
                    'aide_version': fs_version
                },
                'Database': {
                    'version': dbVersion,
                    'details': dbInfo
                }
            }