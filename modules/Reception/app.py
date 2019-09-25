'''
    The Reception module handles project overviews
    and the like.

    2019 Benjamin Kellenberger
'''

import os
import cgi
from bottle import request, response, static_file, redirect, SimpleTemplate
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

    
    def loginCheck(self, project=None, admin=False, superuser=False, extend_session=False):
        return self.login_check(project, admin, superuser, extend_session)

    
    def addLoginCheckFun(self, loginCheckFun):
        if not self.demoMode:
            self.login_check = loginCheckFun


    def _initBottle(self):

        ''' general AIde routings '''
        @self.app.route('/favicon.ico')
        def favicon():
            return static_file('favicon.ico', root=os.path.join(self.staticDir, 'img'))

        @self.app.route('/about')
        def about():
            return static_file('about.html', root=os.path.join(self.staticDir, 'templates'))

        @self.app.route('/')
        def index():
            # redirect to project overview page
            if self.loginCheck():
                return redirect('/projects')
            else:
                return static_file('index.html', root=os.path.join(self.staticDir, 'templates'))


        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projects.html')), 'r') as f:
            self.proj_template = SimpleTemplate(f.read())

        @self.app.route('/projects')
        def projects():

            username = 'Demo mode' if self.demoMode else cgi.escape(request.get_cookie('username'))
            return self.proj_template.render(username=username)


        @self.app.get('/getProjects')
        def get_projects():
            
            username = cgi.escape(request.get_cookie('username'))
            isSuperUser = self.loginCheck(superuser=True)

            projectInfo = self.middleware.get_project_info(username, isSuperUser)
            return {'projects': projectInfo}
