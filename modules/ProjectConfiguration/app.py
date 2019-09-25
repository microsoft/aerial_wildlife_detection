'''
    Bottle routings for the ProjectConfigurator web frontend,
    handling project setup, data import requests, etc.

    2019 Benjamin Kellenberger
'''

import os
import cgi
from urllib.parse import urljoin
import bottle
from bottle import request, response, static_file, redirect, abort, SimpleTemplate
from .backend.middleware import ProjectConfigMiddleware


class ProjectConfigurator:

    def __init__(self, config, app):

        self.config = config
        self.app = app
        self.staticDir = 'modules/ProjectConfiguration/static'
        self.middleware = ProjectConfigMiddleware(config)

        self.login_check = None

        self._initBottle()
    

    def loginCheck(self, project=None, admin=False, superuser=False, extend_session=False):
        return self.login_check(project, admin, superuser, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    def _initBottle(self):


        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectConfiguration.html')), 'r') as f:
            self.projConf_template = SimpleTemplate(f.read())


        @self.app.route('/config/static/<filename:re:.*>')
        def send_static(filename):
            return static_file(filename, root=self.staticDir)


        @self.app.route('/<project>/configuration')
        def configuration_page(project):
            if self.loginCheck(project=project, admin=True):
                username = cgi.escape(request.get_cookie('username'))

                projData = self.middleware.getProjectInfo(project)
                projData['username'] = username

                response = self.projConf_template.render(**projData)
                # response = self.projConf_template.render(
                #     username=username,
                #     projectTitle=projData['projectTitle'],
                #     projectDescr=projData['projectDescr'],
                #     numImagesPerBatch=projData['numImagesPerBatch'],
                #     minImageWidth=projData['minImageWidth']
                # )
                # response.set_header("Cache-Control", "public, max-age=604800")
            else:
                response = bottle.response
                response.status = 303
                response.set_header('Location', '/')
            return response


        # @self.app.post('/getProjectConfiguration')
        # def get_project_configuration():
        #     if not self.loginCheck(admin=True):
        #         abort(401, 'forbidden')

        
        @self.app.post('/<project>/saveProjectConfiguration')
        def save_project_configuration(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')