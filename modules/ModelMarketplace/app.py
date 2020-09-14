'''
    Bottle routings for the model marketplace.
    Handles I/O for sharing a model state (either
    publicly or only within the author's projects)
    and selecting shared states from neighboring
    projects.
    Also supports model state import and export
    to and from the disk, as well as the web.

    2020 Benjamin Kellenberger
'''

import os
import datetime
import tempfile
import uuid
import json
import html
from bottle import static_file, request, response, abort
import requests
from .backend.middleware import ModelMarketplaceMiddleware
from util import helpers


class ModelMarketplace:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self.middleware = ModelMarketplaceMiddleware(config)

        self.login_check = None
        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def __redirect(self, loginPage=False, redirect=None):
        location = ('/login' if loginPage else '/')
        if loginPage and redirect is not None:
            location += '?redirect=' + redirect
        response = bottle.response
        response.status = 303
        response.set_header('Location', location)
        return response


    def _initBottle(self):

        @self.app.get('/<project>/getModelsMarketplace')
        def get_models_marketplace(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                username = html.escape(request.get_cookie('username'))
                modelStates = self.middleware.getModelsMarketplace(project, username)
                return {'modelStates': modelStates}
            except Exception as e:
                return {'status': 1, 'message': str(e)}

        
        @self.app.post('/<project>/importModel')
        def import_model(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                # get data
                username = html.escape(request.get_cookie('username'))
                modelID = request.json['model_id']
                result = self.middleware.importModel(project, username, modelID)
                return result

            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @self.app.post('/<project>/shareModel')
        def share_model(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                # get data
                username = html.escape(request.get_cookie('username'))
                modelID = request.json['model_id']
                modelName = request.json['model_name']
                modelDescription = request.json['model_description']
                public = request.json['public']
                anonymous = request.json['anonymous']

                result = self.middleware.shareModel(project, username,
                                                    modelID, modelName, modelDescription,
                                                    public, anonymous)
                return result
            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @self.app.post('/<project>/unshareModel')
        def unshare_model(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                # get data
                username = html.escape(request.get_cookie('username'))
                modelID = request.json['model_id']

                result = self.middleware.unshareModel(project, username, modelID)
                return result
            except Exception as e:
                return {'status': 1, 'message': str(e)}