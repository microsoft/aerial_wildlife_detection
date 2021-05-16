'''
    Handles administration (sharing, uploading, selecting, etc.)
    of model states through the model marketplace.

    2020-21 Benjamin Kellenberger
'''

import os
import glob
from collections.abc import Iterable
from uuid import UUID
import json
from psycopg2 import sql
import celery
from ai import PREDICTION_MODELS
from modules.LabelUI.backend.middleware import DBMiddleware     # required to obtain label class definitions (TODO: make more elegant)
from . import celery_interface


class ModelMarketplaceMiddleware:

    BUILTIN_MODELS_DIR = 'ai/marketplace'

    MODEL_STATE_REQUIRED_FIELDS = (
        'aide_model_version',
        'name',
        'author',
        'labelclasses',
        'ai_model_library'
    )

    def __init__(self, config, dbConnector, taskCoordinator):
        self.config = config
        self.dbConnector = dbConnector
        self.taskCoordinator = taskCoordinator
        self.labelUImiddleware = DBMiddleware(config, dbConnector)
        self._init_available_ai_models()
        self._load_builtin_model_states()

    

    def _init_available_ai_models(self):
        '''
            Checks the locally installed model implementations
            and retains a list of those that support sharing
            (i.e., the addition of new label classes).
        '''
        self.availableModels = set()
        for key in PREDICTION_MODELS:
            if 'canAddLabelclasses' in PREDICTION_MODELS[key] and \
                PREDICTION_MODELS[key]['canAddLabelclasses'] is True:
                self.availableModels.add(key)



    def _load_builtin_model_states(self):
        '''
            Parses and returns a dict of AI model states built-in
            to AIDE that can be shared on the Model Marketplace.
        '''
        self.builtin_model_states = {}
        modelFiles = glob.glob(os.path.join(self.BUILTIN_MODELS_DIR, '*.json'))
        for modelPath in modelFiles:
            try:
                modelState = json.load(open(modelPath, 'r'))
                for field in self.MODEL_STATE_REQUIRED_FIELDS:
                    if field not in modelState:
                        raise Exception(f'Invalid model state; missing field "{field}".')
                
                # check if model library is installed
                modelLib = modelState['ai_model_library']
                if modelLib not in self.availableModels:
                    raise Exception(f'Model library "{modelLib}"" is not installed in this instance of AIDE.')

                # append to dict
                modelID = 'aide://' + modelPath
                predictionType = modelState['prediction_type']
                if predictionType not in self.builtin_model_states:
                    self.builtin_model_states[predictionType] = {}
                self.builtin_model_states[predictionType][modelID] = {
                    'id': modelID,
                    'name': modelState['name'],
                    'author': modelState['author'],
                    'description': (modelState['description'] if 'description' in modelState else None),
                    'labelclasses': modelState['labelclasses'],
                    'model_library': modelState['ai_model_library'],
                    'annotationType': modelState['annotation_type'],
                    'predictionType': predictionType,
                    'time_created': (modelState['time_created'] if 'time_created' in modelState else None),
                    'alcriterion_library': (modelState['alcriterion_library'] if 'alcriterion_library' in modelState else None),
                    'public': True,
                    'anonymous': False,
                    'selectCount': None,
                    'is_owner': False,
                    'shared': True,
                    'tags': (modelState['tags'] if 'tags' in modelState else None),
                    'citation_info': modelState.get('citation_info', None),
                    'license': modelState.get('license', None),
                    'origin_project': None,
                    'origin_uuid': None,
                    'origin_uri': 'built-in'
                }

            except Exception as e:
                print(f'WARNING: encountered invalid model state "{modelPath}" (message: "{str(e)}").')



    
    def _get_builtin_model_states(self, annotationType, predictionType):
        '''
            Filters the built-in AI model states for provided annotation
            and prediction types and returns a subset accordingly.
        '''
        result = {}
        if predictionType in self.builtin_model_states:
            for modelID in self.builtin_model_states[predictionType].keys():
                modelDict = self.builtin_model_states[predictionType][modelID]
                if modelDict['annotationType'] == annotationType:
                    result[modelDict['id']] = modelDict
        return result

    

    def getModelsMarketplace(self, project, username, modelIDs=None):
        '''
            Returns a dict of model state meta data,
            filtered by the project settings (model library;
            annotation type, prediction type).
            Models can optionally be filtered by an Iterable of
            "modelIDs" (note that built-in states that have not
            yet been added to the database will be appended
            regardless).
        '''

        # get project meta data (immutables, model library)
        projectMeta = self.dbConnector.execute(
            '''
                SELECT annotationType, predictionType, ai_model_library
                FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1
        )
        if projectMeta is None or not len(projectMeta):
            raise Exception(f'Project {project} could not be found in database.')
        projectMeta = projectMeta[0]
        annotationType = projectMeta['annotationtype']
        predictionType = projectMeta['predictiontype']

        # get project models to cross-check which ones have already been imported from the Marketplace
        modelsProject = {}
        projectModelMeta = self.dbConnector.execute(sql.SQL('''
            SELECT marketplace_origin_id, imported_from_marketplace, timeCreated
            FROM {id_cnnstate};
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), None, 'all')
        if projectModelMeta is not None and len(projectModelMeta):
            for model in projectModelMeta:
                if model['marketplace_origin_id'] is not None and model['imported_from_marketplace'] is True:
                    modelsProject[str(model['marketplace_origin_id'])] = model['timecreated']


        queryArgs = [username, project, annotationType, predictionType, project]

        # filter for model IDs if needed
        if isinstance(modelIDs, str):
            modelIDs = (UUID(modelIDs),)
        elif isinstance(modelIDs, UUID):
            modelIDs = (modelIDs,)
        elif isinstance(modelIDs, Iterable):
            modelIDs = list(modelIDs)
            for m in range(len(modelIDs)):
                if not isinstance(modelIDs[m], UUID):
                    modelIDs[m] = UUID(modelIDs[m])

        if modelIDs is not None:
            mIDstr = sql.SQL('AND id IN (%s)')
            queryArgs.append(tuple(modelIDs))
        else:
            mIDstr = sql.SQL('')

        # get matching model states
        result = self.dbConnector.execute(
            sql.SQL('''
                SELECT id, name, description, labelclasses, model_library,
                    annotationType, predictionType, EXTRACT(epoch FROM timeCreated) AS time_created, alcriterion_library,
                    public, anonymous, selectCount,
                    is_owner, shared, tags, citation_info, license,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE author END AS author,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE origin_project END AS origin_project,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE origin_uuid END AS origin_uuid,
                    origin_uri
                FROM (
                    SELECT *,
                    CASE WHEN author = %s AND origin_project = %s THEN TRUE ELSE FALSE END AS is_owner
                    FROM aide_admin.modelMarketplace
                    WHERE annotationType = %s AND
                    predictionType = %s
                    AND (
                        (public = TRUE AND shared = TRUE) OR
                        origin_project = %s
                    )
                    {mIDstr}
                ) AS mm
                LEFT OUTER JOIN (
                    SELECT name AS projectName, shortname
                    FROM aide_admin.project
                ) AS pn
                ON mm.origin_project = pn.shortname;
            ''').format(
                mIDstr=mIDstr
            ),
            tuple(queryArgs),
            'all'
        )
        builtinStates = set()       # built-ins that have already been added to database; no need to add them again
        if result is not None and len(result):
            matchingStates = {}
            for r in result:
                stateID = str(r['id'])
                values = {}
                for key in r.keys():
                    if isinstance(r[key], UUID):
                        values[key] = str(r[key])
                    else:
                        values[key] = r[key]
                
                # check if model has already been imported into project
                if stateID in modelsProject:
                    values['time_imported'] = modelsProject[stateID].timestamp()
                matchingStates[stateID] = values

                originURI = r['origin_uri']
                if isinstance(originURI, str) and originURI.lower().startswith('aide://'):
                    builtinStates.add(originURI)
        else:
            matchingStates = {}

        # augment with built-in model states
        builtins = self._get_builtin_model_states(annotationType, predictionType)
        for key in builtins.keys():
            stateID = str(builtins[key]['id'])
            if stateID not in builtinStates:
                matchingStates[stateID] = builtins[key]
            
        # matchingStates = {**matchingStates, **builtins}

        return matchingStates


    
    def getModelIdByName(self, modelName):
        '''
            Returns the ID of a model in the Model Marketplace under
            a given name, or None if it does not exist.
        '''
        return celery_interface.worker.getModelIdByName(modelName)



    def importModelDatabase(self, project, username, modelID):
        '''
            Imports a model that has been shared via the database
            to the current project.
        '''
        process = celery_interface.import_model_database.si(modelID, project, username)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'ModelMarketplace')

        return {
            'status': 0,
            'task_id': taskID
        }



    def importModelURI(self, project, username, modelURI, public=True, anonymous=False, forceReimport=True, namePolicy='skip', customName=None):
        '''
            Tries to retrieve a model from a given URI (either with prefix "aide://" for
            local models or URL for Web imports) and, if successful, adds it to the
            current project.
            Model definitions need to be in correct JSON format.
            Models are always shared on Model Marketplace (meta data retrieved from model
            definition and decoupled from project). Hence, if a model with the same
            modelURI has already been imported before, AIDE will simply skip loading the
            model and import it from the existing database record.
        '''
        process = celery.chain(
            celery_interface.import_model_uri.si(project, username, modelURI, public, anonymous,
                                                    forceReimport, namePolicy, customName),         # first import to Model Marketplace...
            celery_interface.import_model_database.s(project, username)                             # ...and then to the project
        )
        taskID = self.taskCoordinator.submitJob(project, username, process, 'ModelMarketplace')

        return {
            'status': 0,
            'task_id': taskID
        }



    def importModelFile(self, project, username, modelFile, public=True, anonymous=False, namePolicy='skip', customName=None):
        '''
            Receives a file, uploaded by a user, and tries to import it to the Model
            Marketplace.
            This is done in-place (without Celery). Upon successful completion of the
            import, a task is launched to import the model to the project.
            Model definitions need to be in correct JSON format.
            Models are always shared on Model Marketplace (meta data retrieved from model
            definition and decoupled from project). Hence, if a model with the same
            modelURI has already been imported before, AIDE will simply skip loading the
            model and import it from the existing database record.
        '''

        # import model from file
        modelID = celery_interface.worker.importModelFile(project, username, modelFile.file, modelFile.raw_filename, \
                                                            public, anonymous, namePolicy, customName)

        process = celery_interface.import_model_database.s(modelID, project, username)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'ModelMarketplace')

        return {
            'status': 0,
            'task_id': taskID
        }
    


    def shareModel(self, project, username, modelID, modelName, modelDescription, tags,
                    public, anonymous):
        '''
            Shares a model from a given project on the Model Marketplace.
        '''
        process = celery_interface.share_model.si(project, username, modelID, modelName,
                    modelDescription, tags,
                    public, anonymous)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'ModelMarketplace')

        return {
            'status': 0,
            'task_id': taskID
        }



    def reshareModel(self, project, username, modelID):
        '''
            Unlike "shareModel", this checks for a model that had already been
            shared in the past, but then hidden from the marketplace, and simply
            turns the "shared" attribute back on, if the model could be found.
        '''
        # get origin UUID of model and delegate to "shareModel" function
        modelID = self.dbConnector.execute('''
            SELECT origin_uuid FROM aide_admin.modelMarketplace
            WHERE origin_project = %s
            AND author = %s
            AND id = %s;
        ''',
        (project, username, UUID(modelID)), 1)
        if modelID is None or not len(modelID):
            return {
                'status': 2,
                'message': f'Model with ID "{str(modelID)}" could not be found on the Model Marketplace.'
            }
        
        modelID = modelID[0]['origin_uuid']
        return self.shareModel(project, username, modelID, None, None, None, None, None)
    

    
    def unshareModel(self, project, username, modelID):
        '''
            Sets the "shared" flag to False for a given model state,
            if the provided username is the owner of it.
        '''
        if not isinstance(modelID, UUID):
            modelID = UUID(modelID)

        # check if user is authorized
        isAuthorized = self.dbConnector.execute('''
            SELECT shared FROM aide_admin.modelMarketplace
            WHERE author = %s AND id = %s;
        ''', (username, modelID), 1)

        if isAuthorized is None or not len(isAuthorized):
            # model does not exist or else user has no modification rights
            return {
                'status': 2,
                'message': f'Model with id "{str(modelID)}"" does not exist or else user {username} does not have modification rights to it.'
            }

        # unshare
        self.dbConnector.execute('''
            UPDATE aide_admin.modelMarketplace
            SET shared = FALSE
            WHERE author = %s AND id = %s;
        ''', (username, modelID))

        return {'status': 0}



    def requestModelDownload(self, project, username, modelID, source,
                            modelName=None, modelDescription='', modelTags=[]):
        '''
            Dispatches a Celery task that exports a model state from the database
            (or one of the built-in models) to a file, either just an AIDE-compliant
            JSON definition file, or a zip file with JSON definition and binary model
            state dict file in it, if needed.
            Parameter "source" can either be 'marketplace' or 'project' and dictates
            which database table to download the model from.
            Returns a response containing the Celery task ID that corresponds to the
            download job.
        '''
        process = celery_interface.request_model_download.si(project, username, modelID,
                                                        source, modelName,
                                                        modelDescription, modelTags)
        taskID = self.taskCoordinator.submitJob(project, username, process, 'ModelMarketplace')

        return {
            'status': 0,
            'task_id': taskID
        }