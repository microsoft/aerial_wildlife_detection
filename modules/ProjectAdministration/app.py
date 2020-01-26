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
        self.staticDir = 'modules/ProjectAdministration/static'
        self.middleware = ProjectConfigMiddleware(config)

        self.login_check = None

        self._initBottle()
    

    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    def __redirect_login_page(self):
        response = bottle.response
        response.status = 303
        response.set_header('Location', '/login')
        return response

    
    def _initBottle(self):

        # read templates first
        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectConfiguration.html')), 'r') as f:
            self.projConf_template = SimpleTemplate(f.read())
        
        self.panelTemplates = {}
        panelNames = [
            'accessControl',
            'aiModel',
            'dataDownload',
            'dataUpload',
            'general',
            'labelClasses',
            'overview',
            'userInterface',
            'userPerformance'
        ]
        for pn in panelNames:
            with open(os.path.join(self.staticDir, 'templates/panels', pn + '.html'), 'r') as f:
                self.panelTemplates[pn] = SimpleTemplate(f.read())


        @self.app.route('/<project>/config/static/<filename:re:.*>')
        def send_static(project, filename):
            return static_file(filename, root=self.staticDir)

        
        @self.app.route('/<project>/config/panels/<panel>')
        def send_static_panel(project, panel):
            if self.loginCheck(project=project, admin=True):
                try:
                    return self.panelTemplates[panel].render(project=project)
                except:
                    abort(404, 'not found')
                # return static_file(panel + '.html', root=os.path.join(self.staticDir, 'templates/panels'))
            else:
                abort(401, 'forbidden')


        # with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectOverview.html')), 'r') as f:
        #     self.projectOverview_template = SimpleTemplate(f.read())

        # @self.app.route('/<project>')
        # def redirect_project_overview(project):
        #     return redirect(project + '/')


        @self.app.route('/<project>')
        @self.app.route('/<project>/')
        def send_project_overview(project):

            if not self.loginCheck():
                return redirect('/')

            # get project data (and check if project exists)
            projectData = self.middleware.getProjectInfo(project, ['name', 'description', 'interface_enabled', 'demomode'])
            if projectData is None:
                return self.__redirect_login_page()

            if not self.loginCheck(project=project, extend_session=True):
                return redirect('/')

            if not self.loginCheck(project=project, admin=True, extend_session=True):
                return redirect('/' + project + '/interface')

            # render overview template
            username = 'Demo mode' if projectData['demomode'] else cgi.escape(request.get_cookie('username'))

            return self.projConf_template.render(
                    projectShortname=project,
                    projectTitle=projectData['name'],
                    username=username)

            # return self.projectOverview_template.render(username=username,
            #     projectShortname=project,
            #     projectTitle=projectData['name'], projectDescr=projectData['description'])


        # @self.app.route('/<project>/configuration')
        # def configuration_page(project):
        #     if self.loginCheck(project=project, admin=True):

        #         # get project name for UI template
        #         projectData = self.middleware.getProjectInfo(project, 'name')

        #         # response.set_header("Cache-Control", "public, max-age=604800")
        #         return self.projConf_template.render(
        #             projectShortname=project,
        #             projectTitle=projectData['name'],
        #             username=cgi.escape(request.get_cookie('username')))
        #     else:
        #         response = bottle.response
        #         response.status = 303
        #         response.set_header('Location', '/')
        #     return response


        @self.app.get('/<project>/getConfig')
        @self.app.post('/<project>/getConfig')
        def get_project_configuration(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                # parse subset of configuration parameters (if provided)
                try:
                    data = request.json
                    params = data['parameters']
                except:
                    params = None

                projData = self.middleware.getProjectInfo(project, params)
                return { 'settings': projData }
            except:
                abort(400, 'bad request')


        @self.app.post('/<project>/saveProjectConfiguration')
        def save_project_configuration(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                settings = request.json
                isValid = self.middleware.updateProjectSettings(project, settings)
                if isValid:
                    return {'success': isValid}
                else:
                    abort(400, 'bad request')
            except:
                abort(400, 'bad request')


        @self.app.post('/<project>/saveClassDefinitions')
        def save_class_definitions(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                classdef = request.json
                success = self.middleware.updateClassDefinitions(project, classdef)
                if success:
                    return {'success': success}
                else:
                    abort(400, 'bad request')
            except:
                abort(400, 'bad request')

        
        @self.app.post('/<project>/renewSecretToken')
        def renew_secret_token(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                newToken = self.middleware.renewSecretToken(project)
                return {'secret_token': newToken}
            except:
                abort(400, 'bad request')


        @self.app.get('/<project>/getUsers')
        def get_project_users(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            users = self.middleware.getProjectUsers(project)
            return {'users':users}



        ''' Project creation '''
        with open(os.path.abspath(os.path.join('modules/ProjectAdministration/static/templates/newProject.html')), 'r') as f:
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

                return redirect('/{}/configuration'.format(projSettings['shortname']))

            except Exception as e:
                abort(400, str(e))


        @self.app.get('/verifyProjectName')
        def check_project_name_valid():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            
            try:
                projName = cgi.escape(request.query['name'])
                if len(projName):
                    available = self.middleware.getProjectNameAvailable(projName)
                else:
                    available = False
                return { 'available': available }

            except:
                abort(400, 'bad request')

        
        @self.app.get('/verifyProjectShort')
        def check_project_shortname_valid():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            
            try:
                projName = cgi.escape(request.query['shorthand'])
                if len(projName):
                    available = self.middleware.getProjectShortNameAvailable(projName)
                else:
                    available = False
                return { 'available': available }

            except:
                abort(400, 'bad request')