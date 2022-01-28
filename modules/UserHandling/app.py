'''
    Main Bottle and routings for the UserHandling module.

    2019-22 Benjamin Kellenberger
'''

import os
import html
from bottle import request, response, static_file, abort, redirect, SimpleTemplate
from constants.version import AIDE_VERSION
from .backend.middleware import UserMiddleware
from .backend.exceptions import *


class UserHandler():

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app
        self.staticDir = 'modules/UserHandling/static'
        self.middleware = UserMiddleware(config, dbConnector)

        self.indexURI = self.config.getProperty('Server', 'index_uri', type=str, fallback='/')

        self._initBottle()


    def _parse_parameter(self, request, param):
        if not param in request:
            raise ValueMissingException(param)
        return request.get(param)


    def _initBottle(self):

        # read templates
        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/loginScreen.html')), 'r', encoding='utf-8') as f:
            self.loginScreen_template = SimpleTemplate(f.read())
        
        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/newAccountScreen.html')), 'r', encoding='utf-8') as f:
            self.newAccountScreen_template = SimpleTemplate(f.read())


        @self.app.route('/login')
        @self.app.route('/loginScreen')
        @self.middleware.csrf_token
        def show_login_page():
            return self.loginScreen_template.render(
                version=AIDE_VERSION,
                _csrf_token=request.csrf_token
            )
            # return static_file('loginScreen.html', root=os.path.join(self.staticDir, 'templates'))

        
        @self.app.route('/doLogin', method='POST')
        @self.app.route('/<project>/doLogin', method='POST')
        def do_login(project=None):
            self.middleware.csrf_check(request.forms.get('_csrf_token'))

            # check provided credentials
            try:
                username = html.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')

                # check if session token already provided; renew login if correct
                sessionToken = self.middleware.decryptSessionToken(username, request)
                #request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token'))
                if sessionToken is not None:
                    sessionToken = html.escape(sessionToken)

                sessionToken, _, expires = self.middleware.login(username, password, sessionToken)
                
                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                self.middleware.encryptSessionToken(username, response)
                # response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')

                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))
        

        @self.app.route('/loginCheck', method='POST')
        @self.app.route('/<project>/loginCheck', method='POST')
        def loginCheck(project=None):
            try:
                username = request.get_cookie('username')
                if username is None:
                    username = self._parse_parameter(request.forms, 'username')
                username = html.escape(username)

                sessionToken = self.middleware.decryptSessionToken(username, request)
                # sessionToken = html.escape(request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token')))

                _, _, expires = self.middleware.getLoginData(username, sessionToken)
                
                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                self.middleware.encryptSessionToken(username, response)
                # response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')
                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(401, str(e))


        @self.app.route('/logout', method='GET')        
        @self.app.route('/logout', method='POST')
        @self.app.route('/<project>/logout', method='GET')        
        @self.app.route('/<project>/logout', method='POST')
        def logout(project=None):
            try:
                username = html.escape(request.get_cookie('username'))
                sessionToken = self.middleware.decryptSessionToken(username, request)
                self.middleware.logout(username, sessionToken)
                response.set_cookie('username', '', path='/', expires=0)   #, expires=expires, same_site='strict')
                response.set_cookie('session_token', '',
                            httponly=True, path='/', expires=0)
                # self.middleware.encryptSessionToken(username, response)
                # response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')

                # send redirect
                response.status = 303
                response.set_header('Location', self.indexURI)
                return response

            except Exception as e:
                abort(403, str(e))


        @self.app.route('/<project>/getPermissions', method='POST')
        def get_user_permissions(project):
            try:
                try:
                    username = html.escape(request.get_cookie('username'))
                except:
                    username = None
                if not self.checkAuthenticated(project=project):
                    abort(401, 'not permitted')

                return {
                    'permissions': self.middleware.getUserPermissions(project, username)
                }
            except:
                abort(400, 'bad request')


        @self.app.route('/getUserNames', method='POST')
        @self.app.route('/<project>/getUserNames', method='POST')
        def get_user_names(project=None):
            if project is None:
                try:
                    project = request.json['project']
                except:
                    # no project specified (all users); need be superuser for this
                    project = None

            if self.checkAuthenticated(project, admin=True, superuser=(project is None), extend_session=True):
                return {
                    'users': self.middleware.getUserNames(project)
                }

            else:
                abort(401, 'forbidden') 


        @self.app.route('/doCreateAccount', method='POST')
        def createAccount():
            self.middleware.csrf_check(request.forms.get('_csrf_token'), regenerate=False)

            #TODO: make secret token match
            try:
                username = html.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')
                email = html.escape(self._parse_parameter(request.forms, 'email'))

                sessionToken, _, expires = self.middleware.createAccount(
                    username, password, email
                )

                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                self.middleware.encryptSessionToken(username, response)
                # response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')
                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))


        @self.app.route('/createAccount')
        @self.middleware.csrf_token
        def showNewAccountPage():
            # check if token is required; if it is and wrong token provided, show login screen instead
            try:
                targetToken = html.escape(self.config.getProperty('UserHandler', 'create_account_token'))
            except:
                # no secret token defined
                targetToken = None
            if targetToken is not None and not(targetToken == ''):
                try:
                    providedToken = html.escape(request.query['t'])
                    if providedToken == targetToken:
                        page = self.newAccountScreen_template.render(
                            version=AIDE_VERSION,
                            _csrf_token=request.csrf_token
                        )
                        # response = static_file('templates/newAccountScreen.html', root=self.staticDir)
                    else:
                        page = redirect('/login')
                except:
                    page = redirect('/login')
            else:
                # no token required
                page = self.newAccountScreen_template.render(
                    version=AIDE_VERSION,
                    _csrf_token=request.csrf_token
                )
                # response = static_file('templates/newAccountScreen.html', root=self.staticDir)
            response.set_header('Cache-Control', 'public, max-age=0')
            return page



        @self.app.route('/accountExists', method='POST')
        def checkAccountExists():
            self.middleware.csrf_check(request.forms.get('_csrf_token'), regenerate=False)
            username = ''
            email = ''
            try:
                username = html.escape(self._parse_parameter(request.forms, 'username'))
            except: pass
            try:
                email = html.escape(self._parse_parameter(request.forms, 'email'))
            except: pass
            try:
                return { 'response': self.middleware.accountExists(username, email) }
            except Exception as e:
                abort(401, str(e))


        @self.app.get('/getAuthentication')
        @self.app.post('/getAuthentication')
        def getAuthentication():
            if not self.checkAuthenticated():
                return { 'authentication': {
                        'canCreateProjects': False,
                        'isSuperUser': False
                    }
                }
            try:
                username = html.escape(request.get_cookie('username'))

                # optional: project
                if 'project' in request.query:
                    project = html.escape(request.query['project'])
                else:
                    project = None

                return { 'authentication': self.middleware.getAuthentication(username, project) }

            except:
                return { 'authentication': {
                        'canCreateProjects': False,
                        'isSuperUser': False
                    }
                }


        @self.app.post('/setPassword')
        def setPassword():
            '''
                Routine for super users to set the password of
                a user.
            '''
            if self.checkAuthenticated(superuser=True):
                try:
                    data = request.json
                    username = data['username']
                    password = data['password']
                    result = self.middleware.setPassword(username, password)
                    return result

                except Exception as e:
                    return {
                        'success': False,
                        'message': str(e)
                    }
            else:
                abort(404, 'not found')


        @self.app.get('/v')
        def userVerify():
            '''
                Reserve for future implementations that require
                unauthorized but token-protected user services.
            '''
            abort(404, 'not found')


    def checkAuthenticated(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False, return_all=False):
        username = None
        sessionToken = None
        try:
            username = html.escape(request.get_cookie('username'))
            sessionToken = self.middleware.decryptSessionToken(username, request)
        except:
            pass
        
        try:
            return self.middleware.isAuthenticated(username, sessionToken, project, admin, superuser, canCreateProjects, extend_session, return_all)
        except:
            return False


    def getLoginCheckFun(self):
        return self.checkAuthenticated