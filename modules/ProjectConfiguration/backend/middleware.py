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
from .db_fields import Fields_annotation, Fields_prediction


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
            'annotationType': result['annotationtype'],
            'predictionType': result['predictiontype'],
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


    def createProject(self, username, properties):
        '''
            Receives the most basic, mostly non-changeable settings for a new project
            ("properties") with the following entries:
            - shortname
            - name
            - description
            - annotationType
            - predictionType

            More advanced settings (UI config, AI model, etc.) will be configured after
            the initial project creation stage.

            Verifies whether these settings are available for a new project. If they are,
            it creates a new database schema for the project, adds an entry for it to the
            admin schema table and makes the current user admin. Returns True in this case.
            Otherwise raises an exception.
        '''

        shortname = properties['shortname']

        # verify availability of the project name and shortname
        if not self.getProjectNameAvailable(properties['name']):
            raise Exception('Project name "{}" unavailable.'.format(properties['name']))
        if not self.getProjectShortNameAvailable(shortname):
            raise Exception('Project shortname "{}" unavailable.'.format(shortname))

        # load base SQL
        with open('modules/ProjectConfiguration/static/sql/create_schema.sql', 'r') as f:
            queryStr = sql.SQL(f.read())

        
        # determine annotation and prediction types and add fields accordingly
        annotationFields = list(getattr(Fields_annotation, properties['annotationType']).value)
        predictionFields = list(getattr(Fields_prediction, properties['predictionType']).value)


        # create project schema
        self.dbConnector.execute(queryStr.format(
                id_schema=sql.Identifier(shortname),
                id_auth=sql.Identifier(self.config.getProperty('Database', 'user')),
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
            None,
            None
        )

        # register project
        self.dbConnector.execute('''
            INSERT INTO aide_admin.project (shortname, name, description,
                secret_token,
                annotationType, predictionType)
            VALUES (
                %s, %s, %s,
                %s,
                %s, %s
            );
            ''',
            (
                shortname,
                properties['name'],
                (properties['description'] if 'description' in properties else ''),
                secrets.token_urlsafe(32),
                properties['annotationType'],
                properties['predictionType'],
            ),
            None)

        # register user in project
        self.dbConnector.execute('''
                INSERT INTO aide_admin.authentication (username, project, isAdmin)
                VALUES (%s, %s, true);
            ''',
            (username, shortname,),
            None)
        
        return True



    def updateProjectSettings(self, username, projectSettings):
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


        #TODO

        return True


    def getProjectNameAvailable(self, projectName):
        '''
            Returns True if the provided project (long) name is available.
        '''

        result = self.dbConnector.execute('''SELECT 1 AS result
            FROM aide_admin.project
            WHERE name = %s;
            ''',
            (projectName,),
            1)
        
        if result is None or not len(result):
            return True
        else:
            return result[0]['result'] != 1


    def getProjectShortNameAvailable(self, projectName):
        '''
            Returns True if the provided project shortname is available.
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