'''
    Middleware layer between the project configuration front-end
    and the database.

    2019 Benjamin Kellenberger
'''

import ast
import secrets
from psycopg2 import sql
from modules.Database.app import Database


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


    def getUserStatistics(self, project):
        '''
            Returns statistics for the users that contributed to the
            project in question.
        '''

        queryStr = sql.SQL('''
        
        ''').format(
            
        )
        pass