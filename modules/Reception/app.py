'''
    The Reception module handles project overviews
    and the like.

    2019 Benjamin Kellenberger
'''

import os
import cgi
from bottle import request, response, static_file, redirect, abort, SimpleTemplate
from .backend.middleware import ReceptionMiddleware


class Reception:

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = 'modules/Reception/static'
        self.middleware = ReceptionMiddleware(config)

        self.demoMode = config.getProperty('Project', 'demoMode', type=bool, fallback=False)    #TODO: project-specific

        self.login_check = None

        self._initBottle()

    
    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)

    
    def addLoginCheckFun(self, loginCheckFun):
        if not self.demoMode:
            self.login_check = loginCheckFun


    def _initBottle(self):

        ''' general AIde routings '''
        @self.app.route('/favicon.ico')
        def favicon():
            return static_file('favicon.ico', root=os.path.join(self.staticDir, 'img'))


        @self.app.route('/static/<filename:re:.*>')
        def send_static(filename):
            return static_file(filename, root=self.staticDir)


        @self.app.route('/about')
        @self.app.route('/<project>/about')
        def about(project=None):
            return static_file('about.html', root=os.path.join(self.staticDir, 'templates'))


        @self.app.route('/login')
        def show_login_page():
            if self.demoMode:
                return redirect('/')
            else:
                return static_file('index.html', root=os.path.join(self.staticDir, 'templates'))



        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projects.html')), 'r') as f:
            self.proj_template = SimpleTemplate(f.read())

        @self.app.route('/')
        def projects():
            try:
                if self.demoMode:
                    username = 'Demo mode'
                elif self.login_check():
                    username = cgi.escape(request.get_cookie('username'))
                else:
                    username = ''
            except:
                username = ''
            return self.proj_template.render(username=username)


        @self.app.get('/getProjects')
        def get_projects(): 
            try:
                if self.login_check():
                    username = cgi.escape(request.get_cookie('username'))
                else:
                    username = ''
            except:
                username = ''
            isSuperUser = self.loginCheck(superuser=True)

            projectInfo = self.middleware.get_project_info(username, isSuperUser)
            return {'projects': projectInfo}


        @self.app.get('/<project>/enroll')
        def enroll_in_project(project):
            '''
                Adds a user to the list of contributors to a project
                if it is set to "public", or else if the secret token
                provided matches.
            '''
            try:
                if not self.login_check():
                    abort(401)
                
                username = cgi.escape(request.get_cookie('username'))

                # try to get secret token
                try:
                    providedToken = cgi.escape(request.query['t'])
                except:
                    providedToken = None

                success = self.middleware.enroll_in_project(project, username, providedToken)
                if not success:
                    abort(401)
                return redirect('/{}/'.format(project))     #TODO: doesn't work
            except Exception as e:
                abort(400)
            pass