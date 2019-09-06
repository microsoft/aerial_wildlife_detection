'''
    Middleware layer between the project configuration front-end
    and the database.

    2019 Benjamin Kellenberger
'''

from modules.Database.app import Database


class ProjectConfigMiddleware:
    
    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    