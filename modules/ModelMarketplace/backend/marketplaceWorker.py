'''
    This class deals with all tasks related to the Model Marketplace,
    including model sharing, im-/export, download preparation, etc.

    2020-21 Benjamin Kellenberger
'''

import os
import io
import shutil
import tempfile
import zipfile
from datetime import datetime
from uuid import UUID
import json
from urllib import request
from psycopg2 import sql

from ai import PREDICTION_MODELS
from modules.Database.app import Database
from modules.AIWorker.backend.fileserver import FileServer
from modules.LabelUI.backend.middleware import DBMiddleware     # required to obtain label class definitions (TODO: make more elegant)
from constants.version import MODEL_MARKETPLACE_VERSION
from util.helpers import current_time, get_class_executable, FILENAMES_PROHIBITED_CHARS


class ModelMarketplaceWorker:

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
        self.tempDir = self.config.getProperty('LabelUI', 'tempfiles_dir', type=str, fallback=tempfile.gettempdir())   #TODO
        self.tempDir = os.path.join(self.tempDir, 'aide/modelDownloads')
        os.makedirs(self.tempDir, exist_ok=True)
        self._init_available_ai_models()

    

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
            raise Exception(f'Model state with id "{str(modelID)}" could not be found in the model marketplace.')
        meta = meta[0]

        # check if model type is registered with AIDE
        modelLib = meta['model_library']
        if modelLib not in self.availableModels:
            raise Exception(f'Model type with identifier "{modelLib}" does not support sharing across project, or else is not registered with this installation of AIDE.')

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
            raise Exception(errors)

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
                if isinstance(defaultOptions, dict):
                    defaultOptions = json.dumps(defaultOptions)
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
        if insertedModelID is None or not len(insertedModelID):
            raise Exception('An error occurred while importing model into project.')
        return {
            'status': 0,
            'id': str(insertedModelID[0]['id'])
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
            # model already exists; import to project
            modelID = modelExists[0]['id']
            return self.importModelDatabase(project, username, modelID)

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
                    modelState = f.read().decode('utf-8')   #TODO: handle composed AIDE models (.zip)
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
            if field == 'ai_model_library':
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
        
        # model import to Marketplace successful; now import to projet
        modelID = success[0]['id']
        return self.importModelDatabase(project, username, modelID)



    def shareModel(self, project, username, modelID, modelName, modelDescription, tags,
                    public, anonymous):
        '''
            Shares a model from a given project on the Model Marketplace.
        '''
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



    def prepareModelDownload(self, project, modelID, username, source='marketplace', modelName=None, modelDescription='', modelTags=[]):
        '''
            Attempts to create a file from a model state, either from the database
            (if "modelID" is a UUID) or directly from one of the built-in configu-
            rations (if it is a str, corresponding to the built-in name).
            Constructs an AIDE Model Marketplace-compliant definition JSON file
            in the process, optionally wrapped together with a state dict binary
            file in a zip file, if needed.
            Saves the file to a temporary directory and returns the file path as
            a Celery result if successful.
            Various parameters like "modelName", "modelDescription", etc., are only
            needed if model state is pulled from the project's table and therefore
            does not automatically come with these metadata.
        '''
        # try to parse modelID as UUID
        try:
            modelID = UUID(modelID)
        except:
            # need to load model from built-ins
            pass
        
        if isinstance(modelID, UUID):
            # load from database
            modelDefinition = {
                'aide_model_version': MODEL_MARKETPLACE_VERSION,
                'name': (modelName if modelName is not None else str(modelID)),
                'description': modelDescription,
                'tags': modelTags
            }

            if source.lower() == 'marketplace':
                # load from marketplace table
                queryStr = '''
                    SELECT name, description, author, timeCreated, tags, labelclasses,
                        annotationType, predictionType, model_library, --TODO: more?
                        stateDict
                    FROM aide_admin.modelMarketplace
                    WHERE id = %s;
                '''
            elif source.lower() == 'project':
                # load from project table
                queryStr = sql.SQL('''
                    SELECT timeCreated, model_library, stateDict
                    FROM {id_cnnstate}
                    WHERE id = %s;
                ''').format(
                    id_cnnstate=sql.Identifier(project, 'cnnstate')
                )
            
            result = self.dbConnector.execute(queryStr, (modelID,), 1)
            if result is None or not len(result):
                raise Exception(f'Model state with id "{str(modelID)}" could not be found in database.')
            result = result[0]
            
            for key in result.keys():
                if key == 'timecreated':
                    modelDefinition['time_created'] = result[key].timestamp()
                elif key == 'labelclasses':
                    modelDefinition[key] = json.loads(result[key])
                elif key == 'tags':
                    modelDefinition[key] = result[key].split(';;')
                elif key == 'model_library':
                    modelDefinition['ai_model_library'] = result[key]
                elif key == 'annotationtype':
                    modelDefinition['annotation_type'] = result[key]
                elif key == 'predictiontype':
                    modelDefinition['prediction_type'] = result[key]
                elif key in modelDefinition:
                    modelDefinition[key] = result[key]
            
            # get model implementation meta data
            modelImplementationID = modelDefinition['ai_model_library']
            if modelImplementationID not in PREDICTION_MODELS:
                raise Exception(f'Model implementation with ID "{modelImplementationID}" is not installed in this instance of AIDE.')
            modelMeta = PREDICTION_MODELS[modelImplementationID]
            if 'annotation_type' not in modelDefinition:
                modelDefinition['annotation_type'] = modelMeta['annotationType']
            if 'prediction_type' not in modelDefinition:
                modelDefinition['prediction_type'] = modelMeta['predictionType']
            if 'labelclasses' not in modelDefinition:
                # query from current project and just append as a list
                queryStr = sql.SQL('SELECT name FROM {} ORDER BY name;').format(sql.Identifier(project, 'labelclass'))
                labelClasses = self.dbConnector.execute(queryStr, (project,), 'all')
                labelClasses = [l['name'] for l in labelClasses]
                modelDefinition['labelclasses'] = labelClasses
            
            # model settings: grab from project if possible
            #TODO

            # state dict
            stateDict = result['statedict']
            
            # prepare temporary output file
            destName = modelDefinition['name'] + '_' + current_time().strftime('%Y-%m-%d_%H-%M-%S')
            for char in FILENAMES_PROHIBITED_CHARS:
                destName = destName.replace(char, '_')
            
            # write contents
            if stateDict is None:
                destName += '.json'
                json.dump(modelDefinition, open(os.path.join(self.tempDir, destName), 'w'))

            else:
                destName += '.zip'
                with zipfile.ZipFile(os.path.join(self.tempDir, destName), 'w', zipfile.ZIP_DEFLATED) as f:
                    f.writestr('modelDefinition.json', json.dumps(modelDefinition))
                    bio = io.BytesIO(stateDict)
                    f.writestr('modelState.bin', bio.getvalue())

            return destName

        else:
            # built-in model; copy to temp dir and return path directly
            sourcePath = modelID.replace('aide://', '').strip('/')
            if not os.path.exists(sourcePath):
                raise Exception(f'Model file "{sourcePath}" could not be found.')

            _, fileName = os.path.split(sourcePath)
            destPath = os.path.join(self.tempDir, fileName)
            if not os.path.exists(destPath):
                shutil.copyfile(sourcePath, destPath)

            return destPath