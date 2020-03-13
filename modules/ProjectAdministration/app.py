'''
    Bottle routings for the ProjectConfigurator web frontend,
    handling project setup, data import requests, etc.
    Also handles creation of new projects.

    2019-20 Benjamin Kellenberger
'''

import os
import html
import json
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
    

    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False, return_all=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session, return_all)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    def __redirect_login_page(self):
        response = bottle.response
        response.status = 303
        response.set_header('Location', '/login')
        return response

    
    def _initBottle(self):

        # read templates first
        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectLandingPage.html')), 'r') as f:
            self.projLandPage_template = SimpleTemplate(f.read())

        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projectConfiguration.html')), 'r') as f:
            self.projConf_template = SimpleTemplate(f.read())
        
        self.panelTemplates = {}
        panelNames = [
            'accessControl',
            'aiModel',
            'dataManagement',
            'dataDownload',
            'dataUpload',
            'general',
            'labelClasses',
            'overview',
            'userInterface',
            'userPerformance',
            'modelPerformance'
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
            else:
                abort(401, 'forbidden')


        @self.app.route('/<project>')
        @self.app.route('/<project>/')
        def send_project_overview(project):

            #TODO: show for public project...
            if not self.loginCheck():
                return redirect('/')

            # get project data (and check if project exists)
            projectData = self.middleware.getProjectInfo(project, ['name', 'description', 'interface_enabled', 'demomode'])
            if projectData is None:
                return self.__redirect_login_page()

            if not self.loginCheck(project=project, extend_session=True):
                return redirect('/')

            # render overview template
            username = 'Demo mode' if projectData['demomode'] else html.escape(request.get_cookie('username'))
            
            return self.projLandPage_template.render(
                projectShortname=project,
                projectTitle=projectData['name'],
                projectDescription=projectData['description'],
                username=username)


        @self.app.route('/<project>/configuration')
        def send_project_config_page(project):
            
            #TODO
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

            # render configuration template
            username = 'Demo mode' if projectData['demomode'] else html.escape(request.get_cookie('username'))

            return self.projConf_template.render(
                    projectShortname=project,
                    projectTitle=projectData['name'],
                    username=username)

        
        @self.app.get('/<project>/getPlatformInfo')
        @self.app.post('/<project>/getPlatformInfo')
        def get_platform_info(project):
            if not self.loginCheck(project, admin=True):
                abort(401, 'forbidden')
            try:
                # parse subset of configuration parameters (if provided)
                try:
                    data = request.json
                    params = data['parameters']
                except:
                    params = None

                projData = self.middleware.getPlatformInfo(project, params)
                return { 'settings': projData }
            except:
                abort(400, 'bad request')

        
        @self.app.get('/<project>/getProjectImmutables')
        @self.app.post('/<project>/getProjectImmutables')
        def get_project_immutables(project):
            if not self.loginCheck(project, admin=True):
                abort(401, 'forbidden')
            return {'immutables': self.middleware.getProjectImmutables(project)}
            

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
                params = request.json
                classdef = params['classes']
                removeMissing = (params['remove_missing'] if 'remove_missing' in params else False)
                if isinstance(classdef, str):
                    # re-parse JSON (might happen in case of single quotes)
                    classdef = json.loads(classdef)
                success = self.middleware.updateClassDefinitions(project, classdef, removeMissing)
                if success:
                    return {'success': success}
                else:
                    abort(400, 'bad request')
            except Exception as e:
                abort(400, str(e))

        
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


        @self.app.get('/<project>/getPermissions')
        def get_project_permissions(project):
            permissions = {
                'can_view': False,
                'can_label': False,
                'is_admin': False
            }

            # project-specific permissions
            config = self.middleware.getProjectInfo(project)
            if config['demomode']:
                permissions['can_view'] = True
                permissions['can_label'] = config['interface_enabled']
            isPublic = config['ispublic']
            if not isPublic and not self.loginCheck(project=project):
                # pretend project does not exist (TODO: suboptimal solution; does not properly hide project from view)
                abort(404, 'not found')

            # user-specific permissions
            userPrivileges = self.loginCheck(project=project, return_all=True)
            permissions['can_view'] = isPublic or userPrivileges['project']['enrolled']
            permissions['can_label'] = config['interface_enabled'] and (userPrivileges['project']['enrolled'])
            permissions['is_admin'] = userPrivileges['project']['isAdmin']
            
            return {'permissions': permissions}



        ''' Project creation '''
        with open(os.path.abspath(os.path.join('modules/ProjectAdministration/static/templates/newProject.html')), 'r') as f:
            self.newProject_template = SimpleTemplate(f.read())

        @self.app.route('/newProject')
        def new_project_page():
            if not self.loginCheck():
                return redirect('/')
            # if not self.loginCheck(canCreateProjects=True):
            #     abort(401, 'forbidden')
            username = html.escape(request.get_cookie('username'))
            return self.newProject_template.render(
                username=username
            )


        @self.app.post('/createProject')
        def create_project():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')

            success = False
            try:
                username = html.escape(request.get_cookie('username'))

                # check provided properties
                projSettings = request.json
                success = self.middleware.createProject(username, projSettings)

            except Exception as e:
                abort(400, str(e))

            if success:
                return {'success':True}
            else:
                abort(500, 'An unknown error occurred.')

            


        @self.app.get('/verifyProjectName')
        def check_project_name_valid():
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'forbidden')
            
            try:
                projName = html.escape(request.query['name'])
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
                projName = html.escape(request.query['shorthand'])
                if len(projName):
                    available = self.middleware.getProjectShortNameAvailable(projName)
                else:
                    available = False
                return { 'available': available }

            except:
                abort(400, 'bad request')