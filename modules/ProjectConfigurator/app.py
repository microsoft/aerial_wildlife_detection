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

        @self.app.post('/getProjectConfiguration')
        def get_project_configuration():
            if not self.loginCheck(True):
                abort(401, 'forbidden')
            

        
        @self.app.post('/saveProjectConfiguration')
        def save_project_configuration():
            if not self.loginCheck(True):
                abort(401, 'forbidden')