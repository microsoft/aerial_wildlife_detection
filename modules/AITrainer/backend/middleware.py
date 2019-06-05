'''
    Middleware for AITrainer: handles requests and updates to and from the database.

    2019 Benjamin Kellenberger
'''

import psycopg2
from util.dbConnection import DBConnector


class DBMiddleware():

    def __init__(self, config):
        self.modelInput = config['AITRAINER']['modelInput']
        if self.modelInput not in (''):
            raise ValueError('{} is not a recognized value for parameter "modelInput".'.format(self.modelInput))
        self.dbConnector = DBConnector(config)

    
    #TODO: methods to get latest interesting images and kick-off training and to upload latest predictions to DB