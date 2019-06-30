'''
    Main Bottle and routings for the LabelUI web frontend.

    2019 Benjamin Kellenberger
'''

import os
import cgi
import bottle
from bottle import request, response, static_file, abort
from .backend.middleware import DBMiddleware


class LabelUI():

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = self.config.getProperty(self, 'staticfiles_dir')
        self.middleware = DBMiddleware(config)

        self.login_check = None

        self._initBottle()


    def loginCheck(self):
        return True if self.login_check is None else self.login_check()


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        ''' static routings '''
        @self.app.route('/')
        def index():
            return static_file("index.html", root=os.path.join(self.staticDir, 'html'))


        @self.app.route('/interface')
        def interface():
            if self.loginCheck():
                response = static_file("interface.html", root=os.path.join(self.staticDir, 'html'))
                response.set_header("Cache-Control", "public, max-age=604800")
            else:
                response = bottle.response
                response.status = 303
                response.set_header('Location', '/')
            return response

        
        @self.app.route('/static/<filename:re:.*>')
        def send_static(filename):
            return static_file(filename, root=self.staticDir)


        ''' dynamic routings '''
        @self.app.get('/getProjectInfo')
        def get_project_info():
            # minimum info (name, description) that can be viewed without logging in
            return {'info': self.middleware.getProjectInfo()}


        @self.app.get('/getProjectSettings')
        def get_project_settings():
            if self.loginCheck():
                settings = {
                    'settings': self.middleware.getProjectSettings()
                }
                return settings
            else:
                abort(401, 'not logged in')


        @self.app.get('/getClassDefinitions')
        def get_class_definitions():
            if self.loginCheck():
                classDefs = {
                    'classes': self.middleware.getClassDefinitions()
                }
                return classDefs
            else:
                abort(401, 'not logged in')


        @self.app.post('/getImages')
        def get_images():
            if self.loginCheck():
                username = cgi.escape(request.get_cookie('username'))
                postData = request.body.read()
                dataIDs = request.json['imageIDs']
                json = self.middleware.getBatch(username, dataIDs)
                return json
            else:
                abort(401, 'not logged in')


        @self.app.get('/getLatestImages')
        def get_latest_images():
            if self.loginCheck():
                username = cgi.escape(request.get_cookie('username'))
                try:
                    limit = int(request.query['limit'])
                except:
                    limit = None
                try:
                    order = int(request.query['order'])
                except:
                    order = 'unlabeled'
                try:
                    subset = int(request.query['subset'])
                except:
                    subset = 'default'  
                json = self.middleware.getNextBatch(username=username, order=order, subset=subset, limit=limit)
                return json
            else:
                abort(401, 'not logged in')


        @self.app.post('/submitAnnotations')
        def submit_annotations():
            if self.loginCheck():
                # parse
                try:
                    username = cgi.escape(request.get_cookie('username'))
                    if username is None:
                        # this should never happen, since we are performing a login check
                        raise Exception('no username provided')
                    submission = request.json
                    status = self.middleware.submitAnnotations(username, submission)
                    return { 'status': status }
                except Exception as e:
                    return {
                        'status': 1,
                        'message': str(e)
                    }
            else:
                abort(401, 'not logged in')




''' Convenience launcher (FOR DEBUGGING ONLY) '''
if __name__ == '__main__':
    
    import argparse
    from runserver import Launcher

    parser = argparse.ArgumentParser(description='Run CV4Wildlife AL Service.')
    parser.add_argument('--instance', type=str, default='LabelUI', const=1, nargs='?')
    args = parser.parse_args()
    Launcher(args)