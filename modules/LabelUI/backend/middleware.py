'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

import psycopg2
from modules.Database.app import Database
from .annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction


class DBMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self._fetchProjectSettings()


    def _fetchProjectSettings(self):
        self.projectSettings = {
            'dataServerURI': self.config.getProperty('LabelUI', 'dataServer_uri'),
            'dataType': self.config.getProperty('Project', 'dataType'),
            'minObjSize': self.config.getProperty('Project', 'minObjSize'),
            'classes': self.getClassDefinitions(),
            'annotationType': self.config.getProperty('LabelUI', 'annotationType'),
            'predictionType': self.config.getProperty('AITrainer', 'annotationType'),
            'numImages_x': self.config.getProperty('LabelUI', 'numImages_x', 3),
            'numImages_y': self.config.getProperty('LabelUI', 'numImages_y', 2),
            'defaultImage_w': self.config.getProperty('LabelUI', 'defaultImage_w', 800),
            'defaultImage_h': self.config.getProperty('LabelUI', 'defaultImage_h', 600),
        }

    def _initSQLstrings(self):
        '''
            Prepares retrieval and submission fragments for SQL queries,
            depending on the kind of annotations.
            TODO: put into dedicated enum?
        '''

    def getProjectSettings(self):
        '''
            Queries the database for general project-specific metadata, such as:
            - Classes: names, indices, default colors
            - Annotation type: one of {class labels, positions, bboxes}
        '''
        return self.projectSettings



    def getClassDefinitions(self):
        '''
            Returns a dictionary with entries for all classes in the project.
        '''
        # query
        schema = self.config.getProperty('Database', 'schema')
        sql = 'SELECT * FROM {}.labelclass'.format(schema)
        classProps, colnames = self.dbConnector.execute(sql, None, None)
        idCol = next(c for c in range(len(colnames)) if colnames[c]=='id')
        classdef = {}
        for c in range(len(classProps)):
            classID = classProps[c][idCol]
            classdef[classID] = dict(zip(colnames, classProps[c]))

        return classdef


    def getAnnotations(self, data):
        '''
            Returns entries from the database based on the list of data entry identifiers specified.
        '''
        #TODO
        print(data)
        pass

    
    def getNextBatch(self, ignoreLabeled=False, limit=None):
        '''
            Returns entries from the database (table 'annotation') according to the following rules:
            - entries are ordered by value in column 'priority' (descending)
            - if 'ignoreLabeled' is set to True, only images without a single associated annotation are returned (may result in an empty set). Otherwise priority is given to unlabeled images, but all images are queried if there are no unlabeled ones left.
            - if 'limit' is a number, the return count will be clamped to it.
        '''
        # prepare returns
        response = {}

        # query
        schema = self.config.getProperty('Database', 'schema')
        sql = '''
            SELECT * FROM (SELECT id AS imageID, filename FROM {}.image) AS img
            LEFT OUTER JOIN ({}) AS pred ON pred.image = img.imageID
            LEFT OUTER JOIN ({}) AS anno ON anno.image = img.imageID
        '''.format(schema,
            getattr(QueryStrings_prediction, self.projectSettings['predictionType']).value.format(schema),
            getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value.format(schema),
            schema)

        if ignoreLabeled:
            sql += '''
             WHERE NOT EXISTS (
                SELECT image FROM {}.annotation AS anno
                WHERE anno.image = img.imageID
            )
            '''.format(schema)
        
        sql += ' ORDER BY pred.priority DESC'
        if limit is not None and isinstance(limit, int):
            sql += ' LIMIT {}'.format(limit)
        sql += ';'

        print(sql)

        nextBatch, colnames = self.dbConnector.execute(sql, None, limit)
        colnameDict = dict(zip(colnames, range(len(colnames))))




        #TODO: optimize this horrible code and provide proper, standardized return names below

        # format and return
        for n in range(len(nextBatch)):
            imgID = nextBatch[n][colnameDict['imageid']]
            if not imgID in response:
                response[imgID] = {
                    'fileName': nextBatch[n][colnameDict['filename']],
                    'predictions': {},
                    'annotations': {}
                }
            predID = nextBatch[n][colnameDict['predid']]
            if predID is not None:
                tokens = [c for c in colnames if c.startswith('pred')]
                predValues = [nextBatch[n][colnameDict[t]] for t in tokens]
                response[imgID]['predictions'][predID] = dict(zip(tokens, predValues))
            annoID = nextBatch[n][colnameDict['annoid']]
            if annoID is not None:
                tokens = [c for c in colnames if c.startswith('anno')]
                annoValues = [nextBatch[n][colnameDict[t]] for t in tokens]
                response[imgID]['annotations'][annoID] = dict(zip(tokens, annoValues))

        return response



        #TODO: gather and standardize returns, return random unlabeled set if no prediction exists
        # ALSO RETUR

        #TODO: implement:
        # 1. Create SQL string
        # 2. Query DB, sanity checks
        # 3. Load images from folder (TODO: better use dedicated image server, needs to be accessible by AITrainer as well...), using image paths defined in DB
        # 4. Call customizable method to reorder or post-process images and labels
        # 5. Return data in standardized format
        

        #TODO: for now we just use a dummy function that returns a random batch of local images
        import os
        import random
        localFiles = os.listdir(self.config.getProperty('FileServer', 'staticfiles_dir'))
        numFiles = (limit if limit is not None else 16)

        response = {}

        # simulate labels
        for s in range(numFiles):
            if self.projectSettings['predictionType'] == 'labels':
                if random.random() > 0.5:
                    userLabel = None
                else:
                    userLabel = random.choice(list(self.projectSettings['classes'].keys()))
                predLabel = random.choice(list(self.projectSettings['classes'].keys()))
                response[str(s)] = {
                    'fileName': localFiles[s],
                    'predictions': {
                        'pred_0': {
                            'label': predLabel,     #TODO: numClasses in settings.ini file?
                            'confidence': random.random(),
                        }
                    },
                    'annotations': {
                        'anno_0': {
                            'label': userLabel
                        }
                    }
                }
            elif self.projectSettings['predictionType'] == 'points':
                response[str(s)] = {
                    'fileName': localFiles[s],
                    'predictions': {
                        'point_0': {
                            'x': 220,
                            'y': 100,
                            'label': random.choice(list(self.projectSettings['classes'].keys())),
                            'confidence': random.random()
                        }
                    },
                    'annotations': {
                        'point_1': {
                            'x': 150,
                            'y': 80,
                            'label': random.choice(list(self.projectSettings['classes'].keys()))
                        }
                    }
                }
            else:
                response[str(s)] = {
                    'fileName': localFiles[s]
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