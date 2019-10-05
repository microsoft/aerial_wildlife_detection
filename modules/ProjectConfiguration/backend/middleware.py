'''
    Middleware layer between the project configuration front-end
    and the database.

    2019 Benjamin Kellenberger
'''

import re
import ast
import secrets
import json
from psycopg2 import sql
from modules.Database.app import Database
from modules.LabelUI.backend.annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction


class ProjectConfigMiddleware:
    
    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    def getProjectInfo(self, project):

        queryStr = sql.SQL('''
        SELECT * FROM aide_admin.project
        WHERE shortname = %s;
        ''')
        result = self.dbConnector.execute(queryStr, (project,), 1)
        result = result[0]

        # parse UI settings
        uiSettings = ast.literal_eval(result['ui_settings'])

        response = {
            'projectTitle': result['name'],
            'projectDescr': result['description'],
            'isPublic': result['ispublic'],
            'secretToken': result['secret_token'],
            'demoMode': result['demomode'],
            'numImagesPerBatch': uiSettings['numImagesPerBatch'],
            'minImageWidth': uiSettings['minImageWidth'],

            'aiModelEnabled': result['ai_model_enabled']
        }
        return response


    def renewSecretToken(self, project):
        '''
            Creates a new secret token, invalidating the old one.
        '''
        try:
            newToken = secrets.token_urlsafe(32)
            result = self.dbConnector.execute('''UPDATE aide_admin.project
                SET secret_token = %s
                WHERE shortname = %s;
                SELECT secret_token FROM aide_admin.project
                WHERE shortname = %s;
            ''', (newToken, project, project,), 1)
            return result[0]['secret_token']
        except:
            # this normally should not happen, since we checked for the validity of the project
            return None


    def getProjectStatistics(self, project):
        '''
            Returns statistics for the project, including:
            - number of images
            - number of views, percentages of labeled images, etc.
            - user statistics
            - TODO
        '''

        queryStr = sql.SQL('''
        
        ''').format(

        )
        pass


    def createProject(self, username, projectSettings):
        '''
            Verifies the provided projectSettings dict, looking for the following fields:
            - shortname
            - name
            - description
            - annotationType
            - predictionType
            - isPublic
            - demoMode
            - ui_settings   (sub-dict; performs further verification of it)
            - TODO

            Also checks if the project shorthand is available.
            If all settings are correct, this sets up a new database schema for the project,
            adds an entry for it to the admin schema table and makes the current user
            admin in the project. Returns True in this case.
            Otherwise raises an exception.
        '''

        # verify provided elements
        fieldNames = [
            'shortname',
            'name',
            'description',
            'secretToken',
            'annotationType',
            'predictionType',
            'isPublic',
            'demoMode',
            'ui_settings'
        ]
        for fn in fieldNames:
            if not fn in projectSettings:
                raise Exception('Missing settings parameter {}.'.format(fn))

        # check UI settings
        fieldNames = [
            'welcomeMessage',
            'numImagesPerBatch',
            'minImageWidth',
            'numImageColumns_max',
            'defaultImage_w',
            'defaultImage_h',
            'styles',       #TODO
            'enableEmptyClass',
            'showPredictions',
            'showPredictions_minConf',
            'carryOverPredictions',
            'carryOverRule',
            'carryOverPredictions_minConf',
            'defaultBoxSize_w',
            'defaultBoxSize_h',
            'minBoxSize_w',
            'minBoxSize_h'
        ]
        uiSettings = projectSettings['ui_settings']
        for fn in fieldNames:
            if not fn in uiSettings:
                raise Exception('Missing UI settings parameter {}.'.format(fn))

        
        # check if project shortname available
        shortname = projectSettings['shortname']
        if not self.getProjectNameAvailable(shortname):
            raise Exception('Project shortname ("{}") is not available.'.format(shortname))


        # load base SQL
        with open('ProjectConfiguration/static/sql/create_schema.sql', 'r') as f:
            queryStr = f.readlines()

        
        # determine annotation and prediction types and add fields accordingly
        fields_ignore = set([
            'id',
            'username',
            'image',
            'meta',
            'timeCreated',
            'timeRequired',
            'unsure',
            'cnnstate',
            'confidence',
            'priority'
        ])
        annotationFields = set(getattr(QueryStrings_annotation, projectSettings['annotationType']))
        annotationFields = list(annotationFields.difference(fields_ignore))
        predictionFields = set(getattr(QueryStrings_prediction, projectSettings['predictionType']))
        predictionFields = list(predictionFields.difference(fields_ignore))


        # create project schema
        self.dbConnector.execute(queryStr.format(
                id_image=sql.Identifier(shortname, 'image'),
                id_iu=sql.Identifier(shortname, 'image_user'),
                id_labelclassGroup=sql.Identifier(shortname, 'labelclassGroup'),
                id_labelclass=sql.Identifier(shortname, 'labelclass'),
                id_annotation=sql.Identifier(shortname, 'annotation'),
                id_cnnstate=sql.Identifier(shortname, 'cnnState'),
                id_prediction=sql.Identifier(shortname, 'prediction'),
                annotation_fields=sql.SQL(', ').join([sql.SQL(field) for field in annotationFields]),
                prediction_fields=sql.SQL(', ').join([sql.SQL(field) for field in predictionFields])
            ),
            (shortname, username,),
            None
        )

        # register project
        self.dbConnector.execute('''
            INSERT INTO aide_admin.project (shortname, name, description,
                secret_token,
                annotationType, predictionType, ui_settings,
                numImages_autoTrain,
                minNumAnnoPerImage,
                maxNumImages_train,
                maxNumImages_inference,
                ai_model_enabled,
                ai_model_library, ai_model_settings,
                ai_alCriterion_library, ai_alCriterion_settings
                )
            VALUES (
                %s, %s, %s,
                %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s,
                %s, %s, %s, %s
            )
            ''',
            (
                shortname,
                projectSettings['projectName'],
                projectSettings['projectDescription'],
                secrets.token_urlsafe(32),
                projectSettings['annotationType'],
                projectSettings['predictionType'],
                json.dumps(projectSettings['uiSettings']),
                projectSettings['numImages_autoTrain'],
                projectSettings['minNumAnnoPerImage'],
                projectSettings['maxNumImages_train'],
                projectSettings['maxNumImages_inference'],
                (projectSettings['modelPath'] is not None),
                projectSettings['modelPath'],
                projectSettings['modelSettings'],
                projectSettings['alCriterionPath'],
                projectSettings['alCriterionSettings']
            ),
            None)

        # register user in project
        self.dbConnector.execute('''
                INSERT INTO aide_admin.authentication (username, project, isAdmin)
                VALUES (%s, %s, true);
            ''',
            (username, shortname,),
            None)

        #TODO

        return True


    def getProjectNameAvailable(self, projectName):
        '''
            Returns True if the provided projectName is available.
            In essence, "available" means that a database schema with the given
            name can be created (this includes Postgres schema name conventions).
            Returns False otherwise.
        '''

        # check first if provided name is valid as per Postgres conventions
        matches = re.findall('(^(pg_|[0-9]).*|.*(\$|\s)+.*)', projectName)
        if len(matches):
            return False

        # check if project shorthand already exists in database
        result = self.dbConnector.execute('''SELECT 1 AS result
            FROM information_schema.schemata
            WHERE schema_name = %s
            UNION ALL
            SELECT 1 FROM aide_admin.project
            WHERE shortname = %s;
            ''',
            (projectName,projectName,),
            2)

        if result is None or not len(result):
            return True

        if len(result) == 2:
            return result[0]['result'] != 1 and result[1]['result'] != 1
        elif len(result) == 1:
            return result[0]['result'] != 1
        else:
            return True