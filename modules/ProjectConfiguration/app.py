'''
    Bottle routings for the ProjectConfigurator web frontend,
    handling project setup, data import requests, etc.

    2019 Benjamin Kellenberger
'''

import os
import html
from urllib.parse import urljoin
import bottle
from bottle import request, response, static_file, redirect, abort, SimpleTemplate


class ProjectConfigurator:

    def __init__(self, config, app):

        self.config = config
        self.app = app
        self.staticDir = 'modules/ProjectConfigurator/static'

        self.login_check = None

        self._initBottle()
    

    def loginCheck(self, needBeAdmin=False):
        return self.login_check(needBeAdmin)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    def _initBottle(self):


        with open(os.path.abspath(os.path.join('modules/ProjectConfiguration/static/templates/projectConfiguration.html')), 'r') as f:
            self.projConf_template = SimpleTemplate(f.read())


        @self.app.route('/config/static/<filename:re:.*>')
        def send_static(filename):
            return static_file(filename, root=self.staticDir)


        @self.app.route('/configuration')
        def configuration_page():
            if self.loginCheck(True):
                username = html.escape(request.get_cookie('username'))
                response = self.projConf_template.render(
                    username=username,
                    projectTitle=self.config.getProperty('Project', 'projectName'),
                    projectDescr=self.config.getProperty('Project', 'projectDescription'),
                    numImagesPerBatch=self.config.getProperty('LabelUI', 'numImagesPerBatch'),
                    minImageWidth=self.config.getProperty('LabelUI', 'minImageWidth')
                )
                # response.set_header("Cache-Control", "public, max-age=604800")
            else:
                response = bottle.response
                response.status = 303
                response.set_header('Location', '/')
            return response


        @self.app.post('/getProjectConfiguration')
        def get_project_configuration():
            if not self.loginCheck(True):
                abort(401, 'forbidden')
            

        
        @self.app.post('/saveProjectConfiguration')
        def save_project_configuration():
            if not self.loginCheck(True):
                abort(401, 'forbidden')