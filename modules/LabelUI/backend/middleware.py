'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

from modules.Database.app import Database
from .annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction, getQueryString, getTableNamesString, getOnConflictString, parseAnnotation


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
        classProps = self.dbConnector.execute(sql, None, 'all')
        classdef = {}
        for c in range(len(classProps)):
            classID = classProps[c]['id']
            classdef[classID] = classProps[c]

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
        # query
        schema = self.config.getProperty('Database', 'schema')
        # sql = '''
        #     SELECT * FROM (SELECT id AS imageID, filename FROM {}.image) AS img
        #     LEFT OUTER JOIN (SELECT {} FROM {}.prediction) AS pred ON pred.image = img.imageID
        #     LEFT OUTER JOIN (SELECT {} FROM {}.annotation) AS anno ON anno.image = img.imageID
        # '''.format(schema,
        #     getQueryString(getattr(QueryStrings_prediction, self.projectSettings['predictionType']).value),
        #     schema,
        #     getQueryString(getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value),
        #     schema)
        sql = '''
            SELECT * FROM (SELECT id AS imageID, filename FROM {}.image) AS img
            LEFT OUTER JOIN (SELECT image, viewcount FROM {}.image_user) AS img_user ON img_user.image = img.imageID
            LEFT OUTER JOIN (SELECT {} FROM {}.prediction) AS pred ON pred.image = img.imageID
            LEFT OUTER JOIN (SELECT {} FROM {}.annotation) AS anno ON anno.image = img.imageID
        '''.format(
            schema, schema,
            getQueryString(getattr(QueryStrings_prediction, self.projectSettings['predictionType']).value),
            schema,
            getQueryString(getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value),
            schema
        )
        if ignoreLabeled:
            sql += '''
                WHERE viewcount = 0 OR viewcount IS NULL
            '''

        # if ignoreLabeled:
        #     sql += '''
        #      WHERE NOT EXISTS (
        #         SELECT image FROM {}.annotation AS anno
        #         WHERE anno.image = img.imageID
        #     )
        #     '''.format(schema)
        
        sql += ' ORDER BY pred.priority DESC'
        # if limit is not None and isinstance(limit, int):
        #     sql += ' LIMIT {}'.format(limit)        #FIXME: this limits the num. annotations, not images!!!
        # sql += ';'

        # if limit is None:
        #     limit = 'all'


        # get cursor
        cursor = self.dbConnector.execute_cursor(sql, None)

        # batch = self.dbConnector.execute(sql, None, limit)

        if limit is not None:
            limit = int(limit)

        # format and return
        response = {}
        while limit is None or len(response) < limit:
            b = cursor.fetchone()
            if b is None:
                break

        # for b in batch:
            imgID = b['imageid']
            if not imgID in response:
                response[imgID] = {
                    'fileName': b['filename'],
                    'predictions': {},
                    'annotations': {}
                }
            pred = {}
            anno = {}
            for key in b.keys():
                if key.startswith('pred'):
                    pred[key.replace('pred','')] = b[key]
                elif key.startswith('anno'):
                    anno[key.replace('anno','')] = b[key]
            if b['predid'] is not None:
                response[imgID]['predictions'][b['predid']] = pred
            if b['annoid'] is not None:
                response[imgID]['annotations'][b['annoid']] = anno

        cursor.close()
        return { 'entries': response }


    def submitAnnotations(self, username, submissions):
        '''
            Sends user-provided annotations to the database.
        '''
        #TODO
        return "not implemented"

        # assemble values
        colnames = []
        values = []
        viewcountValues = []
        for imageKey in submissions['entries']:
            entry = submissions['entries'][imageKey]
            if 'annotations' in entry and len(entry['annotations']):
                for annoKey in entry['annotations']:
                    # assemble annotation values
                    annotation = entry['annotations'][annoKey]
                    nextValues, nextColnames = parseAnnotation(annotation)
                    values.append(nextValues)
                    if not len(colnames) or len(colnames) != len(nextColnames):
                        #TODO: uuugly, this should be standardized somewhere
                        colnames = nextColnames
                    
            viewcountValues.append((username, imageKey,))


        schema = self.config.getProperty('Database', 'schema')

        # annotation table
        sql = '''
            INSERT INTO {}.annotation ({})
            VALUES ( %s )
            ON CONFLICT (id) DO UPDATE SET {};
        '''.format(
            schema,
            getTableNamesString(getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value),
            getOnConflictString(getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value)
        )

        self.dbConnector.insert(sql, values)


        # viewcount table
        sql = '''
            INSERT INTO {}.image_user (username, image, viewcount)
            VALUES ( %s )
            ON CONFLICT (username, image) DO UPDATE SET viewcount = viewcount + 1;
        '''.format(schema)

        self.dbConnector.insert(sql, viewcountValues)


        return 'ok'     #TODO