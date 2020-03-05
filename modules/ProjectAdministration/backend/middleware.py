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
from util.helpers import parse_parameters


class ProjectConfigMiddleware:
    
    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    def getProjectInfo(self, project, parameters):

        # parse parameters (if provided) and compare with mutable entries
        allParams = set([
            'name',
            'description',
            'ispublic',
            'secret_token',
            'demomode',
            'interface_enabled',
            'ui_settings',
            'ai_model_enabled',
            'ai_model_library',
            'ai_model_settings',
            'ai_alcriterion_library',
            'ai_alcriterion_settings',
            'numimages_autotrain',
            'minnumannoperimage',
            'maxnumimages_train'
        ])
        if parameters is not None and parameters != '*':
            if isinstance(parameters, str):
                parameters = [parameters.lower()]
            else:
                parameters = [p.lower() for p in parameters]
            set(parameters).intersection_update(allParams)
        else:
            parameters = allParams
        parameters = list(parameters)
        sqlParameters = ','.join(parameters)

        queryStr = sql.SQL('''
        SELECT {} FROM aide_admin.project
        WHERE shortname = %s;
        ''').format(
            sql.SQL(sqlParameters)
        )
        result = self.dbConnector.execute(queryStr, (project,), 1)
        try:
            result = result[0]
        except:
            print('debug')

        # assemble response
        response = {}
        for param in parameters:
            value = result[param]
            if param == 'ui_settings':
                value = ast.literal_eval(value)
            response[param] = value

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


    def getProjectUsers(self, project):
        '''
            Returns a list of users that are enrolled in the project,
            as well as their roles within the project.
        '''

        queryStr = sql.SQL('SELECT * FROM aide_admin.authentication WHERE project = %s;')
        result = self.dbConnector.execute(queryStr, (project,), 'all')
        return result


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
        with open('modules/ProjectAdministration/static/sql/create_schema.sql', 'r') as f:
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
                id_labelclassGroup=sql.Identifier(shortname, 'labelclassgroup'),
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
                interface_enabled,
                annotationType, predictionType)
            VALUES (
                %s, %s, %s,
                %s,
                %s,
                %s, %s
            );
            ''',
            (
                shortname,
                properties['name'],
                (properties['description'] if 'description' in properties else ''),
                secrets.token_urlsafe(32),
                False,
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



    def updateProjectSettings(self, project, projectSettings):
        '''
            TODO
        '''

        # check UI settings first
        if 'ui_settings' in projectSettings:
            fieldNames = [
                ('welcomeMessage', str),
                ('numImagesPerBatch', int),
                ('minImageWidth', int),
                ('numImageColumns_max', int),
                ('defaultImage_w', int),
                ('defaultImage_h', int),
                ('styles', str),       #TODO
                ('enableEmptyClass', bool),
                ('showPredictions', bool),
                ('showPredictions_minConf', float),
                ('carryOverPredictions', bool),
                ('carryOverRule', str),
                ('carryOverPredictions_minConf', float),
                ('defaultBoxSize_w', int),
                ('defaultBoxSize_h', int),
                ('minBoxSize_w', int),
                ('minBoxSize_h', int)
            ]
            uiSettings_new, uiSettingsKeys_new = parse_parameters(projectSettings['ui_settings'], fieldNames, absent_ok=True, escape=True)   #TODO: escape
            
            # adopt current settings and replace values accordingly
            uiSettings = self.dbConnector.execute('''SELECT ui_settings
                    FROM aide_admin.project
                    WHERE shortname = %s;            
                ''', (project,), 1)
            uiSettings = ast.literal_eval(uiSettings[0]['ui_settings'])
            for kIdx in range(len(uiSettingsKeys_new)):
                uiSettings[uiSettingsKeys_new[kIdx]] = uiSettings_new[kIdx]
            projectSettings['ui_settings'] = json.dumps(uiSettings)


        # parse remaining parameters
        fieldNames = [
            ('description', str),
            ('annotationType', str),
            ('predictionType', str),
            ('isPublic', bool),
            ('secret_token', str),
            ('demoMode', bool),
            ('ui_settings', str)
        ]

        vals, params = parse_parameters(projectSettings, fieldNames, absent_ok=True, escape=True)      #TODO: escape
        vals.append(project)

        # commit to DB
        queryStr = sql.SQL('''UPDATE aide_admin.project
            SET
            {}
            WHERE shortname = %s;
            '''
        ).format(
            sql.SQL(',').join([sql.SQL('{} = %s'.format(item)) for item in params])
        )

        self.dbConnector.execute(queryStr, tuple(vals), None)

        return True



    def updateClassDefinitions(self, project, classdef):
        '''
            Updates the project's class definitions.

            TODO: what if label classes change if there's already annotations/predictions?
        '''
        #TODO
        return False



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