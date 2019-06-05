'''
    Main Bottle and routings for the LabelUI web frontend.

    2019 Amrita Gupta, Benjamin Kellenberger
'''

import os
from bottle import Bottle, static_file


class LabelUI:

    def __init__(self, config):
        self.host = config['LABELUI']['host']
        self.port = config['LABELUI']['port']
        self.staticDir = config['LABELUI']['staticfiles_dir']

        self._initBottle()


    def _initBottle(self):

        app = Bottle()

        ''' static routings '''
        @app.route('/')
        def index():
            # print('INDEX.HTML')
            return static_file("index.html", root=os.path.join(self.staticDir, 'html'))

        @app.route('/<filename:re:.*\.html>')
        def send_html(filename):
            return static_file(filename, root=os.path.join(self.staticDir, 'html'))

        @app.route('/<filename:re:.*\.css>')
        def send_css(filename):
            # print('SOME CSS')
            # print(filename)
            return static_file(filename, root=os.path.join(self.staticDir, 'css'))
            # return bottle.static_file(filename, root=dirname+'/static')

        @app.route('/<filename:re:openseadragon\.min\.js$>')
        def send_osd(filename):
            print('OPENSEADRAGON.MIN.JS')
            print(filename)
            return static_file(filename, root=os.path.join(self.staticDir, 'lib/openseadragon'))
        #     # return bottle.static_file(filename, root=dirname+'/openseadragon')

        @app.route('/<filename:re:image-picker\.min\.js$>')
        def send_imgpkr(filename):
            print('IMAGE-PICKER.MIN.JS')
            print(filename)
            return static_file(filename, root=os.path.join(self.staticDir, 'js'))
        #     # return bottle.static_file(filename, root=dirname+'/openseadragon')

        @app.route('/<filename:re:image-picker-masonry\.js$>')
        def send_imgpkrmasonry(filename):
            print('IMAGE-PICKER-MASONRY.JS')
            print(filename)
            return static_file(filename, root=os.path.join(self.staticDir, 'js'))

        @app.route('/<filename:re:.*_IMG_.*\.JPG$>')
        def send_img(filename):
            print('IMAGE FILE')
            print(filename)
            return static_file(filename, root=os.path.join(self.staticDir, 'img'))


        ''' dynamic routings '''
        #TODO: need the following components:
        # - query latest images (call middleware for that)


        app.run(host=self.host, port=self.port)



''' Convenience for debugging '''
if __name__ == '__main__':
    from configparser import ConfigParser
    config = ConfigParser()
    config.read('config/settings.ini')
    LabelUI(config)