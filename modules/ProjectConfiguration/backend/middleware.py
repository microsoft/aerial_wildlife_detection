'''
    Middleware layer between the project configuration front-end
    and the database.

    2019 Benjamin Kellenberger
'''

import ast
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
            'numImagesPerBatch': uiSettings['numImagesPerBatch'],
            'minImageWidth': uiSettings['minImageWidth']
        }
        return response