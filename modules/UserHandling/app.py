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

        self._initBottle()


    def _parse_parameter(self, request, param):
        if not param in request:
            raise ValueMissingException(param)
        return request.get(param)


    def _initBottle(self):

        @self.app.route('/login', method='POST')
        def login():
            # check provided credentials
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')

                # check if session token already provided; renew login if correct
                sessionToken = request.get_cookie('session_token')
                if sessionToken is not None:
                    sessionToken = cgi.escape(sessionToken)

                sessionToken, _, isAdmin, expires = self.middleware.login(username, password, sessionToken)
                
                response.set_cookie('username', username)   #, expires=expires)
                response.set_cookie('session_token', sessionToken, httponly=True)    #, expires=expires)
                response.set_cookie('isAdmin', ('y' if isAdmin else 'n'), httponly=False)    #, expires=expires)

                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))
        

        @self.app.route('/loginCheck', method='POST')
        def loginCheck():
            try:
                username = request.get_cookie('username')
                if username is None:
                    username = self._parse_parameter(request.forms, 'username')
                username = cgi.escape(username)

                sessionToken = cgi.escape(request.get_cookie('session_token'))

                _, _, isAdmin, expires = self.middleware.getLoginData(username, sessionToken)
                
                response.set_cookie('username', username)   #, expires=expires)
                response.set_cookie('session_token', sessionToken, httponly=True)    #, expires=expires)
                response.set_cookie('isAdmin', ('y' if isAdmin else 'n'), httponly=False)    #, expires=expires)
                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(401, str(e))


        @self.app.route('/logout', method='GET')        
        @self.app.route('/logout', method='POST')
        def logout():
            try:
                username = cgi.escape(request.get_cookie('username'))
                sessionToken = cgi.escape(request.get_cookie('session_token'))
                self.middleware.logout(username, sessionToken)

                response.set_cookie('username', '', expires=0)
                response.set_cookie('session_token', '', expires=0, httponly=True)
                response.set_cookie('isAdmin', '', expires=0, httponly=False)    #, expires=expires)

                # send redirect
                response.status = 303
                response.set_header('Location', '/')
                return response

            except Exception as e:
                abort(403, str(e))


        @self.app.route('/createAccount', method='POST')
        def createAccount():
            #TODO: make secret token match
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
                password = self._parse_parameter(request.forms, 'password')
                email = cgi.escape(self._parse_parameter(request.forms, 'email'))

                sessionToken, _, expires = self.middleware.createAccount(
                    username, password, email
                )

                response.set_cookie('username', username)   #, expires=expires)
                response.set_cookie('session_token', sessionToken, httponly=True)    #, expires=expires)
                response.set_cookie('isAdmin', 'n', httponly=False)    #, expires=expires)
                return {
                    'expires': expires.strftime('%H:%M:%S')
                }

            except Exception as e:
                abort(403, str(e))

        @self.app.route('/createAccountScreen')
        def showNewAccountPage():
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
            try:
                username = cgi.escape(self._parse_parameter(request.forms, 'username'))
                if len(username) == 0:
                    raise Exception('invalid request.')
                return { 'response': self.middleware.accountExists(username) }
            except Exception as e:
                abort(401, str(e))

        @self.app.route('/checkAuthenticated', method='POST')
        def checkAuthenticated():
            try:
                if self.checkAuthenticated():
                    return True
                else:
                    raise Exception('not authenticated.')
            except Exception as e:
                abort(401, str(e))
            return response


    def checkAuthenticated(self, admin=False):
        try:
            username = cgi.escape(request.get_cookie('username'))
            sessionToken = cgi.escape(request.get_cookie('session_token'))
            return self.middleware.isAuthenticated(username, sessionToken, admin)
        except Exception:
            return False


    def getLoginCheckFun(self):
        return self.checkAuthenticated