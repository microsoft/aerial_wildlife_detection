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

from util.helpers import parse_boolean
import uuid
import html
from bottle import static_file, request, abort
from .backend.middleware import ModelMarketplaceMiddleware
from .backend.marketplaceWorker import ModelMarketplaceWorker
from util.cors import enable_cors
from util.helpers import parse_boolean


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


    def _initBottle(self):

        @self.app.get('/<project>/getModelsMarketplace')
        def get_models_marketplace(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                username = html.escape(request.get_cookie('username'))

                try:
                    modelIDs = request.params.get('model_ids')
                    if isinstance(modelIDs, str) and len(modelIDs):
                        modelIDs = modelIDs.split(',')
                    else:
                        modelIDs = None
                except Exception:
                    modelIDs = None

                modelStates = self.middleware.getModelsMarketplace(project, username, modelIDs)
                return {'modelStates': modelStates}
            except Exception as e:
                return {'status': 1, 'message': str(e)}

        
        @self.app.get('/<project>/getModelMarketplaceNameAvailable')
        def get_model_marketplace_name_available(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                model_name = request.params.get('name')

                available = (self.middleware.getModelIdByName(model_name) is None)
                return {'status': 0, 'available': available}
            except Exception as exc:
                return {'status': 1, 'message': str(exc)}


        @self.app.post('/<project>/importModel')
        def import_model(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                # get data
                username = html.escape(request.get_cookie('username'))

                if request.json is not None:

                    model_id = str(request.json['model_id'])
                    try:
                        model_id = uuid.UUID(model_id)

                        # model_id is indeed a UUID; import from database
                        return self.middleware.importModelDatabase(project, username, model_id)

                    except Exception:
                        # model comes from network
                        public = request.json.get('public', True)
                        anonymous = request.json.get('anonymous', False)

                        name_policy = request.json.get('name_policy', 'skip')
                        custom_name = request.json.get('custom_name', None)

                        force_reimport = not model_id.strip().lower().startswith('aide://')

                        return self.middleware.importModelURI(project, username, model_id, public,
                                                                anonymous, force_reimport,
                                                                name_policy, custom_name)

                else:
                    # file upload
                    file = request.files.get(list(request.files.keys())[0])
                    public = parse_boolean(request.params.get('public', True))
                    anonymous = parse_boolean(request.params.get('anonymous', False))

                    name_policy = request.params.get('name_policy', 'skip')
                    custom_name = request.params.get('custom_name', None)

                    return self.middleware.importModelFile(project, username, file, public,
                                                                anonymous, name_policy, custom_name)

            except Exception as exc:
                return {'status': 1, 'message': str(exc)}


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
                modelDescription = request.json.get('model_description', '')
                tags = request.json.get('tags', [])
                citationInfo = request.json.get('citation_info', None)
                license = request.json.get('license', None)
                public = request.json.get('public', True)
                anonymous = request.json.get('anonymous', False)
                result = self.middleware.shareModel(project, username,
                                                    modelID, modelName, modelDescription, tags,
                                                    citationInfo, license,
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