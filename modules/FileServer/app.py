'''
    Serves files, such as images, from a local directory.

    2019-21 Benjamin Kellenberger
'''

import os
from bottle import static_file
from util.cors import enable_cors
from util import helpers


class FileServer():

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app

        if verbose_start:
            print('FileServer'.ljust(helpers.LogDecorator.get_ljust_offset()), end='')

        if not helpers.is_fileServer(config):
            if verbose_start:
                helpers.LogDecorator.print_status('fail')
            raise Exception('Not a valid FileServer instance.')
        
        try:
            self.staticDir = self.config.getProperty('FileServer', 'staticfiles_dir')
            self.staticAddressSuffix = self.config.getProperty('FileServer', 'staticfiles_uri_addendum', type=str, fallback='').strip()

            self._initBottle()
        except Exception as e:
            if verbose_start:
                helpers.LogDecorator.print_status('fail')
            raise Exception(f'Could not launch FileServer (message: "{str(e)}").')

        if verbose_start:
            helpers.LogDecorator.print_status('ok')


    def _initBottle(self):

        ''' static routing to files '''
        # @enable_cors
        # @self.app.route(os.path.join(self.staticAddressSuffix, '<path:path>'))
        # def send_file_deprecated(path):
        #     return static_file(path, root=self.staticDir)

        
        @enable_cors
        @self.app.route(os.path.join('/', self.staticAddressSuffix, '/<project>/files/<path:path>'))
        def send_file(project, path):
            return static_file(path, root=os.path.join(self.staticDir, project))