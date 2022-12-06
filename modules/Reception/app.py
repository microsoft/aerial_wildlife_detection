'''
    The Reception module handles project overviews
    and the like.

    2019-21 Benjamin Kellenberger
'''

import os
import html
from bottle import request, redirect, abort, SimpleTemplate, HTTPResponse
from constants.version import AIDE_VERSION
from util import helpers
from .backend.middleware import ReceptionMiddleware


class Reception:

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app
        self.staticDir = 'modules/Reception/static'
        self.middleware = ReceptionMiddleware(config, dbConnector)
        self.login_check = None

        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False, return_all=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session, return_all)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/projects.html')), 'r', encoding='utf-8') as f:
            self.proj_template = SimpleTemplate(f.read())

        @self.app.route('/')
        def projects():
            try:
                username = html.escape(request.get_cookie('username'))
            except Exception:
                username = ''
            return self.proj_template.render(
                version=AIDE_VERSION,
                username=username
            )


        @self.app.get('/getCreateAccountUnrestricted')
        def get_create_account_unrestricted():
            '''
                Responds True if there's no token required for creating
                an account, else False.
            '''
            try:
                token = self.config.getProperty('UserHandler', 'create_account_token', type=str, fallback=None)
                return {'response': token is None or token == ''}
            except Exception:
                return {'response': False}


        @self.app.get('/getProjects')
        def get_projects():
            try:
                if self.loginCheck():
                    username = html.escape(request.get_cookie('username'))
                else:
                    username = ''
            except Exception:
                username = ''
            is_super_user = self.loginCheck(superuser=True)
            archived = helpers.parse_boolean(request.params.get('archived', None))
            project_info = self.middleware.get_project_info(
                                username, is_super_user,
                                archived)
            return {'projects': project_info}


        @self.app.get('/<project>/enroll/<token>')
        def enroll_in_project(project, token):
            '''
                Adds a user to the list of contributors to a project
                if it is set to "public", or else if the secret token
                provided matches.
            '''
            try:
                if not self.loginCheck():
                    return redirect(f'/login?redirect={project}/enroll/{token}')
                
                username = html.escape(request.get_cookie('username'))

                # # try to get secret token
                # try:
                #     providedToken = html.escape(request.query['t'])
                # except Exception:
                #     providedToken = None

                success = self.middleware.enroll_in_project(project, username, token)
                if not success:
                    abort(401)
                return redirect(f'/{project}/interface')
            except HTTPResponse as res:
                return res
            except Exception as e:
                print(e)
                abort(400)


        @self.app.get('/<project>/getSampleImages')
        def get_sample_images(project):
            '''
                Returns a list of URLs for images in the project,
                if the project is public or the user is enrolled
                in it. Used for backdrops on the project landing
                page.
                Prioritizes golden question images.
            '''

            # check visibility of project
            permissions = self.loginCheck(project=project, return_all=True)

            if not (permissions['project']['isPublic'] or \
                permissions['project']['enrolled'] or \
                permissions['project']['demoMode']):
                abort(401, 'unauthorized')
            
            try:
                limit = int(request.params.get('limit'))
            except Exception:
                limit = 128
            imageURLs = self.middleware.getSampleImages(project, limit)
            return {'images': imageURLs}