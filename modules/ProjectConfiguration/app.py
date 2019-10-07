'''
    Bottle routings for the ProjectConfigurator web frontend,
    handling project setup, data import requests, etc.
    Also handles creation of new projects.

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
    

    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    def _initBottle(self):


        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectConfiguration.html')), 'r') as f:
            self.projConf_template = SimpleTemplate(f.read())


        @self.app.route('/<project>/config/static/<filename:re:.*>')
        def send_static(project, filename):
            return static_file(filename, root=self.staticDir)


        @self.app.route('/<project>/configuration')
        def configuration_page(project):
            if self.loginCheck(project=project, admin=True):

                # get project name for UI template
                projectData = self.middleware.getProjectInfo(project)

                # response.set_header("Cache-Control", "public, max-age=604800")
                return self.projConf_template.render(
                    projectShortname=project,
                    projectTitle=projectData['projectTitle'],
                    username=cgi.escape(request.get_cookie('username')))
            else:
                response = bottle.response
                response.status = 303
                response.set_header('Location', '/')
            return response


        @self.app.get('/<project>/getConfig')
        def get_project_configuration(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                projData = self.middleware.getProjectInfo(project)
                return { 'settings': projData }
            except:
                abort(400, 'bad request')


        @self.app.post('/<project>/saveProjectConfiguration')
        def save_project_configuration(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')



        ''' Project creation '''
        with open(os.path.abspath(os.path.join('modules/ProjectConfiguration/static/templates/newProject.html')), 'r') as f:
            self.newProject_template = SimpleTemplate(f.read())

        @self.app.route('/newProject')
        def new_project_page():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            username = cgi.escape(request.get_cookie('username'))
            return self.newProject_template.render(
                username=username
            )


        @self.app.post('/createProject')
        def create_project():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')

            try:
                username = cgi.escape(request.get_cookie('username'))

                # check provided properties
                projSettings = request.json

                response = self.middleware.createProject(username, projSettings)
                if not response:
                    raise Exception('An unknown error occurred during project creation.')

                return redirect('/{}'.format(projSettings['shortname']))

            except Exception as e:
                abort(400, str(e))


        @self.app.get('/verifyProjectName')
        def check_project_name_valid():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            
            try:
                projName = request.query['name']
                available = self.middleware.getProjectNameAvailable(projName)

                return { 'available': available }

            except:
                abort(400, 'bad request')

        
        @self.app.get('/verifyProjectShort')
        def check_project_shortname_valid():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            
            try:
                projName = request.query['shorthand']
                available = self.middleware.getProjectShortNameAvailable(projName)

                return { 'available': available }

            except:
                abort(400, 'bad request')