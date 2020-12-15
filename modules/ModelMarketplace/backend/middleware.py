'''
    Handles administration (sharing, uploading, selecting, etc.)
    of model states through the model marketplace.

    2020 Benjamin Kellenberger
'''

import os
import glob
from datetime import datetime
from uuid import UUID
from urllib import request
import json
from psycopg2 import sql
from ai import PREDICTION_MODELS
from modules.Database.app import Database
from modules.LabelUI.backend.middleware import DBMiddleware     # required to obtain label class definitions (TODO: make more elegant)
from modules.AIWorker.backend.fileserver import FileServer
from util.helpers import get_class_executable, current_time


class ModelMarketplaceMiddleware:

    BUILTIN_MODELS_DIR = 'ai/marketplace'

    MODEL_STATE_REQUIRED_FIELDS = (
        'aide_model_version',
        'name',
        'author',
        'labelclasses',
        'ai_model_library'
    )

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
        self.labelUImiddleware = DBMiddleware(config)
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

    

    def getModelsMarketplace(self, project, username):
        '''
            Returns a dict of model state meta data,
            filtered by the project settings (model library;
            annotation type, prediction type).
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
            SELECT marketplace_origin_id, timeCreated
            FROM {id_cnnstate};
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), None, 'all')
        if projectModelMeta is not None and len(projectModelMeta):
            for model in projectModelMeta:
                if model['marketplace_origin_id'] is not None:
                    modelsProject[str(model['marketplace_origin_id'])] = model['timecreated']


        # get matching model states
        result = self.dbConnector.execute(
            '''
                SELECT id, name, description, labelclasses, model_library,
                    annotationType, predictionType, EXTRACT(epoch FROM timeCreated) AS time_created, alcriterion_library,
                    public, anonymous, selectCount,
                    is_owner, shared, tags,
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
                ) AS mm
                LEFT OUTER JOIN (
                    SELECT name AS projectName, shortname
                    FROM aide_admin.project
                ) AS pn
                ON mm.origin_project = pn.shortname;
            ''',
            (username, project, annotationType, predictionType, project),
            'all'
        )
        if result is not None and len(result):
            matchingStates = {}
            builtinStates = set()       # built-ins that have already been added to database; no need to add them again
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



    def importModelDatabase(self, project, username, modelID):
        '''
            Imports a model that has been shared via the database
            to the current project.
        '''
        if not isinstance(modelID, UUID):
            modelID = UUID(modelID)

        # get model meta data
        meta = self.dbConnector.execute('''
            SELECT id, name, description, labelclasses, model_library,
            annotationType, predictionType, timeCreated, alcriterion_library,
            public, anonymous, selectCount,
            CASE WHEN anonymous THEN NULL ELSE author END AS author,
            CASE WHEN anonymous THEN NULL ELSE origin_project END AS origin_project,
            CASE WHEN anonymous THEN NULL ELSE origin_uuid END AS origin_uuid
            FROM aide_admin.modelMarketplace
            WHERE id = %s;
        ''', (modelID,), 1)

        if meta is None or not len(meta):
            return {
                'status': 2,
                'message': f'Model state with id "{str(modelID)}" could not be found in the model marketplace.'
            }
        meta = meta[0]

        # check if model type is registered with AIDE
        modelLib = meta['model_library']
        if modelLib not in self.availableModels:
            return {
                'status': 3,
                'message': f'Model type with identifier "{modelLib}" does not support sharing across project, or else is not registered with this installation of AIDE.'
            }

        # check if model hasn't already been imported to current project
        modelExists = self.dbConnector.execute(sql.SQL('''
            SELECT id
            FROM {id_cnnstate}
            WHERE marketplace_origin_id = %s;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), (modelID,), 1)
        if modelExists is not None and len(modelExists):
            # model already exists in project; update timestamp to make it current (TODO: find a better solution)
            self.dbConnector.execute(sql.SQL('''
                UPDATE {id_cnnstate}
                SET timeCreated = NOW()
                WHERE marketplace_origin_id = %s;
            ''').format(
                id_cnnstate=sql.Identifier(project, 'cnnstate')
            ), (modelID))
            return {
                'status': 0,
                'message': 'Model had already been imported to project and was elevated to be the most current.'
            }

        # check if model is suitable for current project
        immutables = self.labelUImiddleware.get_project_immutables(project)
        errors = ''
        if immutables['annotationType'] != meta['annotationtype']:
            meta_at = meta['annotationtype']
            proj_at = immutables['annotationType']
            errors = f'Annotation type of model ({meta_at}) is not compatible with project\'s annotation type ({proj_at}).'
        if immutables['predictionType'] != meta['predictiontype']:
            meta_pt = meta['predictiontype']
            proj_pt = immutables['predictionType']
            errors += f'\nPrediction type of model ({meta_pt}) is not compatible with project\'s prediction type ({proj_pt}).'
        if len(errors):
            return {
                'status': 4,
                'message': errors
            }

        # set project's selected model and options (if different)
        currentLibrary = self.dbConnector.execute('''
            SELECT ai_model_library
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        currentLibrary = currentLibrary[0]['ai_model_library']
        if currentLibrary != meta['model_library']:
            # model library differs; replace existing options with new ones
            try:
                modelClass = get_class_executable(meta['model_library'])
                defaultOptions = modelClass.getDefaultOptions()
            except:
                defaultOptions = None
            self.dbConnector.execute('''
                UPDATE aide_admin.project
                SET ai_model_library = %s,
                ai_model_settings = %s
                WHERE shortname = %s;
            ''', (meta['model_library'], defaultOptions, project))

        # finally, increase selection counter and import model state
        #TODO: - retain existing alCriterion library
        insertedModelID = self.dbConnector.execute(sql.SQL('''
            UPDATE aide_admin.modelMarketplace
            SET selectCount = selectCount + 1
            WHERE id = %s;
            INSERT INTO {id_cnnstate} (marketplace_origin_id, stateDict, timeCreated, partial, model_library, alCriterion_library)
            SELECT id, stateDict, timeCreated, FALSE, model_library, alCriterion_library
            FROM aide_admin.modelMarketplace
            WHERE id = %s
            RETURNING id;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
            ),
            (modelID, modelID),
            1
        )
        return {
            'status': 0,
            'id': str(insertedModelID)
        }



    def importModelURI(self, project, username, modelURI):
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
        assert isinstance(modelURI, str), 'Incorrect model URI provided.'
        warnings = []

        # check if model has already been imported into model marketplace
        modelExists = self.dbConnector.execute('''
            SELECT id
            FROM aide_admin.modelMarketplace
            WHERE origin_uri = %s;
        ''', (modelURI,), 1)
        if modelExists is not None and len(modelExists):
            # model already exists
            result = self.importModelDatabase(project, username, modelExists[0]['id'])
            result['warnings'] = ['Model with identical origin URI found in Model Marketplace and has been imported from there instead.']
            return result

        # check import type
        if modelURI.lower().startswith('aide://'):
            # local import
            localPath = modelURI.replace('aide://', '').strip('/')
            if not os.path.exists(localPath):
                raise Exception(f'Local file could not be found ("{localPath}").')
            with open(localPath, 'r') as f:
                modelState = f.read()      #TODO: raise if not text file

        else:
            # network import
            try:
                with request.urlopen(modelURI) as f:
                    modelState = f.read().decode('utf-8')   #TODO: raise if not text data
            except Exception as e:
                raise Exception(f'Error retrieving model state from URL ("{modelURI}"). Message: "{str(e)}".')

        try:
            modelState = json.loads(modelState)
        except Exception as e:
            raise Exception(f'Model state is not a valid AIDE JSON file (message: "{str(e)}").')

        # project metadata
        projectMeta = self.dbConnector.execute('''
            SELECT annotationType, predictionType
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        if projectMeta is None or not len(projectMeta):
            raise Exception(f'Project with shortname "{project}" not found in database.')
        projectMeta = projectMeta[0]

        # check fields
        for field in self.MODEL_STATE_REQUIRED_FIELDS:
            if field not in modelState:
                raise Exception(f'Missing field "{field}" in AIDE JSON file.')
            if field == 'ai_model_libray':
                # check if model library is installed
                modelLibrary = modelState[field]
                if modelLibrary not in PREDICTION_MODELS:
                    raise Exception(f'Model library "{modelLibrary}" is not installed in this instance of AIDE.')
                # check if annotation and prediction types match
                if projectMeta['annotationtype'] not in PREDICTION_MODELS[modelLibrary]['annotationType']:
                    raise Exception('Project\'s annotation type is not compatible with this model state.')
                if projectMeta['predictiontype'] not in PREDICTION_MODELS[modelLibrary]['predictionType']:
                    raise Exception('Project\'s prediction type is not compatible with this model state.')
        
        # check if model state URI provided
        if hasattr(modelState, 'ai_model_state_uri'):
            stateDictURI = modelState['ai_model_state_uri']
            try:
                if stateDictURI.lower().startswith('aide://'):
                    # load from disk
                    stateDictPath = stateDictURI.replace('aide://', '').strip('/')
                    if not os.path.isfile(stateDictPath):
                        raise Exception(f'Model state file path provided ("{stateDictPath}"), but file could not be found.')
                    with open(stateDictPath, 'rb') as f:
                        stateDict = f.read()        #TODO: BytesIO instead
                
                else:
                    # network import
                    with request.urlopen(stateDictURI) as f:
                        stateDict = f.read()        #TODO: progress bar; load in chunks; etc.

            except Exception as e:
                raise Exception(f'Model state URI provided ("{stateDictURI}"), but could not be loaded (message: "{str(e)}").')
        else:
            stateDict = None
        
        # remaining parameters
        modelName = modelState['name']
        modelAuthor = modelState['author']
        modelDescription = (modelState['description'] if 'description' in modelState else None)
        modelTags = (';;'.join(modelState['tags']) if 'tags' in modelState else None)
        labelClasses = modelState['labelclasses']       #TODO: parse?
        if not isinstance(labelClasses, str):
            labelClasses = json.dumps(labelClasses)
        modelOptions = (modelState['ai_model_settings'] if 'ai_model_settings' in modelState else None)
        modelLibrary = modelState['ai_model_library']
        alCriterion_library = (modelState['alcriterion_library'] if 'alcriterion_library' in modelState else None)
        annotationType = PREDICTION_MODELS[modelLibrary]['annotationType']      #TODO
        predictionType = PREDICTION_MODELS[modelLibrary]['predictionType']      #TODO
        if not isinstance(annotationType, str):
            annotationType = ','.join(annotationType)
        if not isinstance(predictionType, str):
            predictionType = ','.join(predictionType)
        timeCreated = (modelState['time_created'] if 'time_created' in modelState else None)
        try:
            timeCreated = datetime.fromtimestamp(timeCreated)
        except:
            timeCreated = current_time()

        # try to launch model with data
        try:
            modelClass = get_class_executable(modelLibrary)
            modelClass(project=project,
                        config=self.config,
                        dbConnector=self.dbConnector,
                        fileServer=FileServer(self.config).get_secure_instance(project),
                        options=modelOptions)
            
            # verify options
            if modelOptions is not None:
                optionMeta = modelClass.verifyOptions(modelOptions)
                #TODO: parse warnings and errors

        except Exception as e:
            raise Exception(f'Model from imported state could not be launched (message: "{str(e)}").')


        # import model state into Model Marketplace
        success = self.dbConnector.execute('''
            INSERT INTO aide_admin.modelMarketplace
                (name, description, tags, labelclasses, author, statedict,
                model_library, alCriterion_library,
                annotationType, predictionType,
                timeCreated,
                origin_project, origin_uuid, origin_uri, public, anonymous)
            VALUES %s
            ON CONFLICT(origin_uri) DO NOTHING
            RETURNING id;
        ''',
        [(
            modelName, modelDescription, modelTags, labelClasses, modelAuthor,
            stateDict, modelLibrary, alCriterion_library, annotationType, predictionType,
            timeCreated,
            None, None, modelURI, True, False
        )],
        1)
        if success is None or not len(success):
            raise Exception('Model could not be imported into Model Marketplace.')
        
        # also import into project
        result = self.importModelDatabase(project, username, success[0]['id'])
        result['warnings'] = warnings
        return result
    


    def shareModel(self, project, username, modelID, modelName, modelDescription, tags,
                    public, anonymous):
        #TODO: export as Celery task

        if not isinstance(modelID, UUID):
            modelID = UUID(modelID)
        
        if tags is None:
            tags = ''
        elif isinstance(tags, list) or isinstance(tags, tuple):
            tags = ';;'.join(tags)
        
        # check if model class supports sharing
        isShareable = self.dbConnector.execute('''
            SELECT ai_model_library
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        if isShareable is None or not len(isShareable):
            return {
                'status': 1,
                'message': f'Project {project} could not be found in database.'
            }
        modelLib = isShareable[0]['ai_model_library']
        if modelLib not in self.availableModels:
            return {
                'status': 2,
                'message': f'The model with id "{modelLib}" does not support sharing, or else is not installed in this instance of AIDE.'
            }

        # check if model hasn't already been shared
        isShared = self.dbConnector.execute('''
            SELECT id, author, shared
            FROM aide_admin.modelMarketplace
            WHERE origin_project = %s AND origin_uuid = %s;
        ''', (project, modelID), 'all')
        if isShared is not None and len(isShared):
            if not isShared[0]['shared']:
                # model had been shared in the past but then hidden; unhide
                self.dbConnector.execute('''
                    UPDATE aide_admin.modelMarketplace
                    SET shared = True
                    WHERE origin_project = %s AND origin_uuid = %s;
                ''', (project, modelID))
            
            # update shared model meta data
            self.dbConnector.execute('''
                UPDATE aide_admin.modelMarketplace
                SET name = %s,
                description = %s,
                public = %s,
                anonymous = %s,
                tags = %s
                WHERE id = %s AND author = %s;
            ''', (modelName, modelDescription, public, anonymous, tags,
                isShared[0]['id'], username), None)
            return {'status': 0}

        # check if model hasn't been imported from the marketplace
        isImported = self.dbConnector.execute(sql.SQL('''
            SELECT marketplace_origin_id
            FROM {id_cnnstate}
            WHERE id = %s;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ),
        (modelID,),
        1)
        if isImported is not None and len(isImported):
            marketplaceID = isImported[0]['marketplace_origin_id']
            if marketplaceID is not None:
                return {
                    'status': 4,
                    'message': f'The selected model is already shared through the marketplace (id "{str(marketplaceID)}").'
                }

        # get project immutables
        immutables = self.labelUImiddleware.get_project_immutables(project)

        # get label class info
        labelclasses = json.dumps(self.labelUImiddleware.getClassDefinitions(project, False))

        # check if name is unique (TODO: required?)
        nameTaken = self.dbConnector.execute('''
            SELECT COUNT(*) AS cnt
            FROM aide_admin.modelMarketplace
            WHERE name = %s;
        ''', (modelName,), 'all')
        if nameTaken is not None and len(nameTaken) and nameTaken[0]['cnt']:
            return {
                'status': 5,
                'message': f'A model state with name "{modelName}" already exists in the Model Marketplace.'
            }

        # share model state
        sharedModelID = self.dbConnector.execute(sql.SQL('''
            INSERT INTO aide_admin.modelMarketplace
            (name, description, tags, labelclasses, author, statedict,
            model_library, alCriterion_library,
            annotationType, predictionType,
            origin_project, origin_uuid, public, anonymous)

            SELECT %s, %s, %s, %s, %s, statedict,
            model_library, alCriterion_library,
            %s, %s,
            %s, id, %s, %s
            FROM {id_cnnstate} AS cnnS
            WHERE id = %s
            RETURNING id;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), (modelName, modelDescription, tags, labelclasses, username,
            immutables['annotationType'], immutables['predictionType'],
            project, public, anonymous, modelID))

        return {
            'status': 0,
            'shared_model_id': str(sharedModelID)
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