'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

import psycopg2
from util.dbConnection import DBConnector


class DBMiddleware():

    def __init__(self, config):
        self.dbConnector = DBConnector(config)

    
    def getNextBatch(self, ignoreLabeled=True, limit=None):
        '''
            Returns entries from the database (table 'annotation') according to the following rules:
            - entries are ordered by value in column 'priority' (descending)
            - if 'ignoreLabeled' is set to True, only images without a single associated annotation are returned (may result in an empty set). Otherwise priority is given to unlabeled images, but all images are queried if there are no unlabeled ones left.
            - if 'limit' is a number, the return count will be clamped to it.
        '''
        #TODO: implement:
        # 1. Create SQL string
        # 2. Query DB, sanity checks
        # 3. Load images from folder (TODO: better use dedicated image server, needs to be accessible by AITrainer as well...), using image paths defined in DB
        # 4. Call customizable method to reorder or post-process images and labels
        # 5. Return data in standardized format
        pass


    def submitAnnotations(self, imageIDs, annotations, metadata=None):
        '''
            Sends user-provided annotations to the database. Inputs:
            - imageIDs: [N] list of image UUIDs the labels are associated with
            - annotations: [N] list of annotation data (str) for the images
            - metadata: TODO
            - TODO: user session data...
        '''
        #TODO: implement:
        # 1. Sanity checks
        # 2. Create SQL string
        # 3. Submit data to DB
        # 4. Return statistics
        # TODO: need customizable method here too?
        pass