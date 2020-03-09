'''
    Bottle routings for data administration
    (i.e., image, annotation, and prediction
    down- and uploads).
    Needs to be run from instance responsible
    for serving files (i.e., FileServer module).

    2020 Benjamin Kellenberger
'''

import os
import datetime
import tempfile
from bottle import static_file, request, response, abort
import requests
from .backend.middleware import DataAdministrationMiddleware
from util.cors import enable_cors
from util import helpers


class DataAdministrator:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self.is_fileServer = helpers.is_fileServer(config)  # set up either direct methods or relaying
        self.middleware = DataAdministrationMiddleware(config)

        self.login_check = None
        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun

    
    @staticmethod
    def _parse_range(params, paramName, minValue, maxValue):
        '''
            Parses "params" (dict) for a given keyword
            "paramName" (str), and expects a dict with
            keywords "min" and "max" there. One of the
            two may be missing, in which case the values
            of "minValue" and "maxValue" will be used.
            Returns a tuple of (min, max) values, or None
            if "paramName" is not in "params."
        '''
        if params is None or not paramName in params:
            return None
        entry = params[paramName].copy()
        if not 'min' in entry:
            entry['min'] = minValue
        if not 'max' in entry:
            entry['max'] = maxValue
        return (entry['min'], entry['max'])


    def relay_request(self, project, fun):
        '''
            Used to forward requests to the FileServer instance,
            if it happens to be a different machine.
            Requests cannot go directly to the FileServer due to
            CORS restrictions.
        '''
        if self.is_fileServer:
            return None
        else:
            # forward request to FileServer
            cookies = request.cookies.dict
            for key in cookies:
                cookies[key] = cookies[key][0]
            files = {}
            if len(request.files):
                for key in request.files:
                    files[key] = (request.files[key].raw_filename, request.files[key].file, request.files[key].content_type)
            return requests.post(os.path.join(self.config.getProperty('Server', 'dataServer_uri'), project, fun),
                    cookies=cookies, json=request.json, files=files,
                    headers={'User-Agent': 'Mozilla/5.0'})



    def _initBottle(self):

        ''' Status polling '''
        @self.app.post('/<project>/pollStatus')
        def pollStatus(project):
            '''
                Receives a task ID and polls the middleware
                for an ongoing data administration task.
                Returns a dict with (meta-) data, including
                the Celery status type, result (if completed),
                error message (if failed), etc.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                taskID = request.json['taskID']
                status = self.middleware.pollStatus(project, taskID)

                return {'response': status}

            except Exception as e:
                abort(400, str(e))


        ''' Image management functionalities '''
        @enable_cors
        @self.app.post('/<project>/listImages')
        def listImages(project):
            '''
                Returns a list of images and various properties
                and statistics (id, filename, viewcount, etc.),
                all filterable by date and value ranges.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            # parse parameters
            now = helpers.current_time()
            params = request.json

            imageAddedRange = self._parse_range(params, 'imageAddedRange',
                                            datetime.time.min,
                                            now)
            lastViewedRange = self._parse_range(params, 'lastViewedRange',
                                            datetime.time.min,
                                            now)
            viewcountRange = self._parse_range(params, 'viewcountRange',
                                            0,
                                            1e9)
            numAnnoRange = self._parse_range(params, 'numAnnoRange',
                                            0,
                                            1e9)
            numPredRange = self._parse_range(params, 'numPredRange',
                                            0,
                                            1e9)
            orderBy = (params['orderBy'] if 'orderBy' in params else None)
            order = (params['order'].lower() if 'order' in params else None)
            limit = (params['limit'] if 'limit' in params else None)


            # get images
            result = self.middleware.listImages(project,
                                            imageAddedRange,
                                            lastViewedRange,
                                            viewcountRange,
                                            numAnnoRange,
                                            numPredRange,
                                            orderBy,
                                            order,
                                            limit)
            
            return {'response': result}


        @enable_cors
        @self.app.post('/<project>/uploadImages')
        def uploadImages(project):
            '''
                Upload image files through UI.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            if not self.is_fileServer:
                return self.relay_request(project, 'uploadImages')

            try:
                images = request.files
                result = self.middleware.uploadImages(project, images)
                return {'result': result}
            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @self.app.get('/<project>/scanForImages')
        @enable_cors
        def scanForImages(project):
            '''
                Search project file directory on disk for
                images that are not registered in database.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            result = self.middleware.scanForImages(project)
            return {'response': result}


        @enable_cors
        @self.app.post('/<project>/addExistingImages')
        def addExistingImages(project):
            '''
                Add images that exist in project file directory
                on disk, but are not yet registered in database.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                imageNames = request.json
                status, result = self.middleware.addExistingImages(project, imageNames['images'])
                return {'status': status, 'response': result}
            except Exception as e:
                return {'status': 1, 'message': str(e)}


        @enable_cors
        @self.app.post('/<project>/removeImages')
        def removeImages(project):
            '''
                Remove images from database, including predictions
                and annotations (if flag is set).
                Also remove images from disk (if flag is set).
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                data = request.json
                imageIDs = data['images']
                if 'forceRemove' in data:
                    forceRemove = data['forceRemove']
                else:
                    forceRemove = False
                if 'deleteFromDisk' in data:
                    deleteFromDisk = data['deleteFromDisk']
                else:
                    deleteFromDisk = False
                
                images_deleted = self.middleware.removeImages(project,
                                                            imageIDs,
                                                            forceRemove,
                                                            deleteFromDisk)
                
                return {'status': 0, 'images_deleted': images_deleted}

            except Exception as e:
                return {'status': 1, 'message': str(e)}


        ''' Annotation and prediction up- and download functionalities '''
        @enable_cors
        @self.app.get('/<project>/getValidImageExtensions')
        def getValidImageExtensions(project=None):
            '''
                Returns the file extensions for images currently
                supported by AIDE.
            '''
            return {'extensions': helpers.valid_image_extensions}

        
        @enable_cors
        @self.app.get('/<project>/getValidMIMEtypes')
        def getValidMIMEtypes(project=None):
            '''
                Returns the MIME types for images currently
                supported by AIDE.
            '''
            return {'MIME_types': helpers.valid_image_mime_types}

        
        # data download
        @enable_cors
        @self.app.post('/<project>/requestDownload')
        def requestDownload(project):
            '''
                Parses request parameters and then assembles project-
                related metadata (annotations, predictions, etc.) by
                storing them as files on the server in a temporary
                directory.
                Returns the download links to those temporary files.
            '''
            #TODO: allow download for non-admins?
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            # parse parameters
            try:
                params = request.json
                dataType = params['dataType']
                if 'dateRange' in params:
                    dateRange = params['dateRange']
                else:
                    dateRange = None
                if 'users' in params:
                    userList = params['users']
                else:
                    userList = None
                
                taskID = self.middleware.prepareDataDownload(project,
                                                            dataType,
                                                            userList,
                                                            dateRange)
                return {'response': taskID}

            except Exception as e:
                abort(401, str(e))
        

        @enable_cors
        @self.app.route('/<project>/downloadData/<filename:re:.*>')
        def downloadData(project, filename):
            #TODO: allow download for non-admins?
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
                abort(401, 'forbidden')

            if not self.is_fileServer:
                return self.relay_request(project, os.path.join('downloadData', project, filename))
            
            return static_file(filename, root=os.path.join(tempfile.gettempdir(), 'aide/downloadRequests', project), download=True)