'''
    Bottle routings for the model marketplace.
    Handles I/O for sharing a model state (either
    publicly or only within the author's projects)
    and selecting shared states from neighboring
    projects.
    Also supports model state import and export
    to and from the disk, as well as the web.

    2020-21 Benjamin Kellenberger
'''

import uuid
import html
import bottle
from bottle import static_file, request, abort
from .backend.middleware import ModelMarketplaceMiddleware
from .backend.marketplaceWorker import ModelMarketplaceWorker
from util.cors import enable_cors


class ModelMarketplace:

    def __init__(self, config, app, dbConnector, taskCoordinator, verbose_start=False):
        self.config = config
        self.app = app

        self.middleware = ModelMarketplaceMiddleware(config, dbConnector, taskCoordinator)
        self.tempDir = ModelMarketplaceWorker(self.config, dbConnector).tempDir

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
                try:
                    modelID = uuid.UUID(modelID)

                    # modelID is indeed a UUID; import from database
                    return self.middleware.importModelDatabase(project, username, modelID)

                except:
                    # model comes from file or network
                    return self.middleware.importModelURI(project, username, modelID)

            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @self.app.post('/<project>/requestModelDownload')
        def request_model_download(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                # get data
                username = html.escape(request.get_cookie('username'))
                modelID = request.json['model_id']
                source = request.json['source']
                assert source in ('marketplace', 'project'), 'invalid download source provided'

                # optional values (for project-specific model download)
                modelName = (request.json['model_name'] if 'model_name' in request.json else None)
                modelDescription = (request.json['model_description'] if 'model_description' in request.json else '')
                modelTags = (request.json['model_tags'] if 'model_tags' in request.json else [])

                result = self.middleware.requestModelDownload(project, username,
                                                            modelID, source,
                                                            modelName, modelDescription, modelTags)
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
                tags = request.json['tags']
                public = request.json['public']
                anonymous = request.json['anonymous']
                result = self.middleware.shareModel(project, username,
                                                    modelID, modelName, modelDescription, tags,
                                                    public, anonymous)
                return result
            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @self.app.post('/<project>/reshareModel')
        def reshare_model(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                # get data
                username = html.escape(request.get_cookie('username'))
                modelID = request.json['model_id']

                result = self.middleware.reshareModel(project, username, modelID)
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


        @enable_cors
        @self.app.route('/<project>/download/models/<filename:re:.*>')
        def download_model(project, filename):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
                abort(401, 'forbidden')

            return static_file(filename, root=self.tempDir, download=True)