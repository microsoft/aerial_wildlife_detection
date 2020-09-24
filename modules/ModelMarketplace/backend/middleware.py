'''
    Handles administration (sharing, uploading, selecting, etc.)
    of model states through the model marketplace.

    2020 Benjamin Kellenberger
'''

from uuid import UUID
import json
from psycopg2 import sql
from ai import PREDICTION_MODELS
from modules.Database.app import Database
from modules.LabelUI.backend.middleware import DBMiddleware     # required to obtain label class definitions (TODO: make more elegant)


class ModelMarketplaceMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self.labelUImiddleware = DBMiddleware(config)
    

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

        # get matching model states
        result = self.dbConnector.execute(
            '''
                SELECT id, name, description, labelclasses, model_library,
                    annotationType, predictionType, EXTRACT(epoch FROM timeCreated) AS time_created, alcriterion_library,
                    public, anonymous, selectCount,
                    is_owner, shared,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE author END AS author,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE origin_project END AS origin_project,
                    CASE WHEN NOT is_owner AND anonymous THEN NULL ELSE origin_uuid END AS origin_uuid
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
            for r in result:
                stateID = str(r['id'])
                values = {}
                for key in r.keys():
                    if isinstance(r[key], UUID):
                        values[key] = str(r[key])
                    else:
                        values[key] = r[key]
                matchingStates[stateID] = values
        else:
            matchingStates = {}

        return matchingStates



    def importModel(self, project, username, modelID):

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
        if modelLib not in PREDICTION_MODELS:
            return {
                'status': 3,
                'message': f'Model type with identifier "{modelLib}" is not registered with this installation of AIDE.'
            }

        # check if model hasn't already been imported to current project
        modelExists = self.dbConnector.execute(sql.SQL('''
            SELECT id
            FROM {id_cnnstate}
            WHERE marketplace_origin_id = %s;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), (modelID), 1)
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

        # set project's selected model
        self.dbConnector.execute('''
            UPDATE aide_admin.project
            SET ai_model_library = %s
            WHERE shortname = %s;
        ''', (meta['model_library'], project))

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


    
    def shareModel(self, project, username, modelID, modelName, modelDescription,
                    public, anonymous):
        #TODO: export as Celery task

        if not isinstance(modelID, UUID):
            modelID = UUID(modelID)

        # check if model hasn't already been shared
        isShared = self.dbConnector.execute('''
            SELECT author, shared
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
                return {'status': 0}
            else:
                # model has been shared and still is
                author = isShared[0]['author']
                return {
                    'status': 2,
                    'message': f'Model state has already been shared by {author}.'
                }

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
                    'status': 3,
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
                'status': 4,
                'message': f'A model state with name "{modelName}" already exists in the Model Marketplace.'
            }

        sharedModelID = self.dbConnector.execute(sql.SQL('''
            INSERT INTO aide_admin.modelMarketplace
            (name, description, labelclasses, author, statedict,
            model_library, alCriterion_library,
            annotationType, predictionType,
            origin_project, origin_uuid, public, anonymous)

            SELECT %s, %s, %s, %s, statedict,
            model_library, alCriterion_library,
            %s, %s,
            %s, id, %s, %s
            FROM {id_cnnstate} AS cnnS
            WHERE id = %s
            RETURNING id;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate')
        ), (modelName, modelDescription, labelclasses, username,
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
        return self.shareModel(project, username, modelID, None, None, None, None)
    

    
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