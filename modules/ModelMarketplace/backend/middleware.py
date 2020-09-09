'''
    Handles administration (sharing, uploading, selecting, etc.)
    of model states through the model marketplace.

    2020 Benjamin Kellenberger
'''

from uuid import UUID
import json
from psycopg2 import sql
from modules.Database.app import Database
from modules.LabelUI.backend.middleware import DBMiddleware     # required to obtain label class definitions (TODO: make more elegant)


class ModelMarketplaceMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self.labelUImiddleware = DBMiddleware(config)
    

    def getModelsMarketplace(self, project):
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
                SELECT * FROM (
                    SELECT id, name, description, labelclasses, model_library,
                    annotationType, predictionType, EXTRACT(epoch FROM timeCreated) AS time_created, alcriterion_library,
                    public, anonymous, selectCount,
                    CASE WHEN anonymous THEN NULL ELSE author END AS author,
                    CASE WHEN anonymous THEN NULL ELSE origin_project END AS origin_project,
                    CASE WHEN anonymous THEN NULL ELSE origin_uuid END AS origin_uuid
                    FROM aide_admin.modelMarketplace
                    WHERE annotationType = %s AND
                    predictionType = %s
                    AND (
                        public = TRUE OR
                        origin_project = %s
                    )
                ) AS mm
                LEFT OUTER JOIN (
                    SELECT name AS projectName, shortname
                    FROM aide_admin.project
                ) AS pn
                ON mm.origin_project = pn.shortname;
            ''',
            (annotationType, predictionType, project),
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



    def importModel(self, project, modelID):

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
        ''', (UUID(modelID),), 1)

        # check if model is suitable for current project
        #TODO
        return 0


    
    def shareModel(self, project, username, modelID, modelName, modelDescription,
                    public, anonymous):
        #TODO: export as Celery task

        modelID = UUID(modelID)

        # check if model hasn't already been shared
        isShared = self.dbConnector.execute('''
            SELECT author
            FROM aide_admin.modelMarketplace
            WHERE origin_project = %s AND origin_uuid = %s;
        ''', (project, modelID), 'all')
        if isShared is not None and len(isShared):
            author = isShared[0]['author']
            return {
                'status': 2,
                'message': f'Model state has already been shared by {author}.'
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
                'status': 3,
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
            project, public, anonymous, UUID(modelID)))

        return {
            'status': 0,
            'shared_model_id': sharedModelID
        }