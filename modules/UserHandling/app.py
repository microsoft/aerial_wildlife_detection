'''
    Main Bottle and routings for the UserHandling module.

    2019 Benjamin Kellenberger
'''

from bottle import request
from .backend.middleware import UserMiddleware
from .backend.exceptions import *


class UserHandler():

    def __init__(self, config, app):
        self.config = config
        self.app = app
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
                username = self._parse_parameter(request.forms, 'username')
                password = self._parse_parameter(request.forms, 'password')

                sessionToken, timestamp = self.middleware.login(username, password)
                
                response.set_cookie('session_token', sessionToken)  #TODO: expires?
                response.set_cookie('last_login', timestamp)

            except:
                return 'error'  #TODO
        

        @self.app.route('/loginCheck')
        def loginCheck():
            pass