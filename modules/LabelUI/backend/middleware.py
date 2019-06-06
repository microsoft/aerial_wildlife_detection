'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

from collections import namedtuple
import psycopg2
from modules.Database.app import Database


class DBMiddleware():

    ClassEntry = namedtuple('Class', ['name', 'index', 'color'])

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self._getProjectSettings()


    def _getProjectSettings(self):
        '''
            Queries the database for general project-specific metadata, such as:
            - Classes: names, indices, default colors
            - Annotation type: one of {class labels, positions, bboxes}
        '''

        #TODO: dummy for now; implement to query db
        self.projectSettings = {
            'classes': [],
            'annotationType': 'classification'      #TODO: make enum?
        }


    
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
        

        #TODO: for now we just use a dummy function that returns a random batch of local images
        import os
        import numpy as np
        localFiles = os.listdir(self.config.getProperty('FileServer', 'staticfiles_dir'))
        numFiles = (limit if limit is not None else 16)

        selected = np.random.choice(len(localFiles), numFiles, replace=False)

        response = {}

        # simulate labels
        for s in selected:
            if np.random.rand() > 0.5:
                userLabel = None
            else:
                userLabel = np.random.randint(10)
            response[str(s)] = {
                'filePath': localFiles[s],
                'predictedLabel': np.random.randint(10),     #TODO: numClasses in settings.ini file?
                'predictedConfidence': np.random.rand(),
                'userLabel': userLabel
            }
        return response


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
        
        return 'Not yet implemented.'