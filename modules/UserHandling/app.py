'''
    Main Bottle and routings for the UserHandling module.

    2019 Benjamin Kellenberger
'''

import cgi
from bottle import request, response, static_file, abort, redirect
from .backend.middleware import UserMiddleware
from .backend.exceptions import *


class UserHandler():

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = 'modules/UserHandling/static'
        self.middleware = UserMiddleware(config)

        self.indexURI = self.config.getProperty('Server', 'index_uri', type=str, fallback='/')
        self.demoMode = config.getProperty('Project', 'demoMode', type=bool, fallback=False)

        self._initBottle()


    def _parse_parameter(self, request, param):
        if not param in request:
            raise ValueMissingException(param)
        return request.get(param)


    def _initBottle(self):

        
        @self.app.route('/doLogin', method='POST')
        @self.app.route('/<project>/doLogin', method='POST')
        def do_login(project=None):
            if self.demoMode:
                return redirect('/')

            # check provided credentials
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')

                # check if session token already provided; renew login if correct
                sessionToken = request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token'))
                if sessionToken is not None:
                    sessionToken = cgi.escape(sessionToken)

                sessionToken, _, expires = self.middleware.login(username, password, sessionToken)
                
                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')

                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))
        

        @self.app.route('/loginCheck', method='POST')
        @self.app.route('/<project>/loginCheck', method='POST')
        def loginCheck(project=None):
            if self.demoMode:
                response.set_cookie('username', 'Demo mode', path='/')   #, expires=expires, same_site='strict')
                return {
                    'expires': '-1' #expires.strftime('%H:%M:%S')
                }

            try:
                username = request.get_cookie('username')
                if username is None:
                    username = self._parse_parameter(request.forms, 'username')
                username = cgi.escape(username)

                sessionToken = cgi.escape(request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token')))

                _, _, expires = self.middleware.getLoginData(username, sessionToken)
                
                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')
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
            if self.demoMode:
                return redirect('/')

            try:
                username = cgi.escape(request.get_cookie('username'))
                sessionToken = cgi.escape(request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token')))
                self.middleware.logout(username, sessionToken)

                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')

                # send redirect
                response.status = 303
                response.set_header('Location', self.indexURI)
                return response

            except Exception as e:
                abort(403, str(e))


        @self.app.route('/<project>/getPermissions', method='POST')
        def get_user_permissions(project):
            if self.demoMode:
                return {
                    'error': 'not allowed in demo mode'
                }
            try:
                username = cgi.escape(request.get_cookie('username'))
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
            if self.demoMode:
                return redirect('/')

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


        @self.app.route('/createAccount', method='POST')
        def createAccount():
            if self.demoMode:
                return redirect('/')

            #TODO: make secret token match
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')
                email = cgi.escape(self._parse_parameter(request.forms, 'email'))

                sessionToken, _, expires = self.middleware.createAccount(
                    username, password, email
                )

                response.set_cookie('username', username, path='/')   #, expires=expires, same_site='strict')
                response.set_cookie('session_token', sessionToken, httponly=True, path='/', secret=self.config.getProperty('Project', 'secret_token'))    #, expires=expires, same_site='strict')
                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))

        @self.app.route('/createAccountScreen')
        def showNewAccountPage():
            if self.demoMode:
                return redirect('/')

            # check if token is required; if it is and wrong token provided, show login screen instead
            targetToken = cgi.escape(self.config.getProperty('UserHandler', 'create_account_token'))
            if targetToken is not None and not(targetToken == ''):
                try:
                    providedToken = cgi.escape(request.query['t'])
                    if providedToken == targetToken:
                        response = static_file('templates/createAccountScreen.html', root=self.staticDir)
                    else:
                        response = static_file('templates/loginScreen.html', root=self.staticDir)
                except:
                    response = static_file('templates/loginScreen.html', root=self.staticDir)
            else:
                # no token required
                response = static_file('templates/createAccountScreen.html', root=self.staticDir)
            response.set_header('Cache-Control', 'public, max-age=0')
            return response

        @self.app.route('/loginScreen')
        def showLoginPage():
            return static_file('templates/loginScreen.html', root=self.staticDir)

        @self.app.route('/accountExists', method='POST')
        def checkAccountExists():
            if self.demoMode:
                return { 'response': { 'username': False, 'email': False } }
            username = ''
            email = ''
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
            except: pass
            try:
                email = cgi.escape(self._parse_parameter(request.forms, 'email'))
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
                username = cgi.escape(request.get_cookie('username'))

                # optional: project
                if 'project' in request.query:
                    project = cgi.escape(request.query['project'])
                else:
                    project = None

                return { 'authentication': self.middleware.getAuthentication(username, project) }

            except:
                return { 'authentication': {
                        'canCreateProjects': False,
                        'isSuperUser': False
                    }
                }


    def checkAuthenticated(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        if self.demoMode:
            return True

        try:
            username = cgi.escape(request.get_cookie('username'))
            sessionToken = cgi.escape(request.get_cookie('session_token', secret=self.config.getProperty('Project', 'secret_token')))
            return self.middleware.isAuthenticated(username, sessionToken, project, admin, superuser, canCreateProjects, extend_session)
        except:
            return False


    def getLoginCheckFun(self):
        return self.checkAuthenticated