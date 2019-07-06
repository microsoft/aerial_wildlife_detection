'''
    Serves files, such as images, from a local directory.

    Note: this is just a convenience module and should only be used for debugging purposes.
          Use a proper file server (e.g. an Apache instance configured with static directories)
          for deployment.

    2019 Benjamin Kellenberger
'''

import os
from bottle import static_file
from util.cors import enable_cors


class FileServer():

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = self.config.getProperty('FileServer', 'staticfiles_dir')
        self.staticAddress = self.config.getProperty('FileServer', 'staticfiles_uri')
        if not self.staticAddress.startswith('/'):
            self.staticAddress = '/' + self.staticAddress

        self._initBottle()


    def _initBottle(self):

        ''' static routing to files '''
        @self.app.route('/cors', method=['OPTIONS', 'GET'])

        @enable_cors
        @self.app.route(os.path.join(self.staticAddress, '<path:path>'))
        def send_file(path):
            return static_file(path, root=self.staticDir)