'''
    Serves files, such as images, from a local directory.

    2019-20 Benjamin Kellenberger
'''

import os
from bottle import static_file
from util.cors import enable_cors
from util import helpers


class FileServer():

    def __init__(self, config, app):
        self.config = config
        self.app = app

        if not helpers.is_fileServer(config):
            raise Exception('Not a valid FileServer instance.')

        self.staticDir = self.config.getProperty('FileServer', 'staticfiles_dir')
        self.staticAddressSuffix = self.config.getProperty('FileServer', 'staticfiles_uri_addendum', type=str, fallback='').strip()

        self._initBottle()


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