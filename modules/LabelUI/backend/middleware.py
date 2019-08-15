'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

from uuid import UUID
from datetime import datetime
import pytz
import dateutil.parser
import json
from modules.Database.app import Database
from .sql_string_builder import SQLStringBuilder
from .annotation_sql_tokens import QueryStrings_annotation, AnnotationParser


class DBMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self._fetchProjectSettings()
        self.sqlBuilder = SQLStringBuilder(config)
        self.annoParser = AnnotationParser(config)


    def _fetchProjectSettings(self):
        # AI controller URI
        aiControllerURI = self.config.getProperty('Server', 'aiController_uri')
        if aiControllerURI is None or aiControllerURI.strip() == '':
            # no AI backend configured
            aiControllerURI = None

        # LabelUI drawing styles
        with open(self.config.getProperty('LabelUI', 'styles_file', type=str, fallback='modules/LabelUI/static/json/styles.json'), 'r') as f:
            styles = json.load(f)

        # Image backdrops for index screen
        with open(self.config.getProperty('Project', 'backdrops_file', type=str, fallback='modules/LabelUI/static/json/backdrops.json'), 'r') as f:
            backdrops = json.load(f)

        # Welcome message for UI tutorial
        with open(self.config.getProperty('Project', 'welcome_message_file', type=str, fallback='modules/LabelUI/static/templates/welcome_message.html'), 'r') as f:
            welcomeMessage = f.readlines()

        self.projectSettings = {
            'projectName': self.config.getProperty('Project', 'projectName'),
            'projectDescription': self.config.getProperty('Project', 'projectDescription'),
            'indexURI': self.config.getProperty('Server', 'index_uri', type=str, fallback='/'),
            'dataServerURI': self.config.getProperty('Server', 'dataServer_uri'),
            'aiControllerURI': aiControllerURI,
            'dataType': self.config.getProperty('Project', 'dataType', fallback='images'),
            'classes': self.getClassDefinitions(),
            'enableEmptyClass': self.config.getProperty('Project', 'enableEmptyClass', fallback='no'),
            'annotationType': self.config.getProperty('Project', 'annotationType'),
            'predictionType': self.config.getProperty('Project', 'predictionType'),
            'showPredictions': self.config.getProperty('LabelUI', 'showPredictions', fallback='yes'),
            'showPredictions_minConf': self.config.getProperty('LabelUI', 'showPredictions_minConf', type=float, fallback=0.5),
            'carryOverPredictions': self.config.getProperty('LabelUI', 'carryOverPredictions', fallback='no'),
            'carryOverRule': self.config.getProperty('LabelUI', 'carryOverRule', fallback='maxConfidence'),
            'carryOverPredictions_minConf': self.config.getProperty('LabelUI', 'carryOverPredictions_minConf', type=float, fallback=0.75),
            'defaultBoxSize_w': self.config.getProperty('LabelUI', 'defaultBoxSize_w', type=int, fallback=10),
            'defaultBoxSize_h': self.config.getProperty('LabelUI', 'defaultBoxSize_h', type=int, fallback=10),
            'minBoxSize_w': self.config.getProperty('Project', 'box_minWidth', type=int, fallback=1),
            'minBoxSize_h': self.config.getProperty('Project', 'box_minHeight', type=int, fallback=1),
            'numImagesPerBatch': self.config.getProperty('LabelUI', 'numImagesPerBatch', type=int, fallback=1),
            'minImageWidth': self.config.getProperty('LabelUI', 'minImageWidth', type=int, fallback=300),
            'numImageColumns_max': self.config.getProperty('LabelUI', 'numImageColumns_max', type=int, fallback=1),
            'defaultImage_w': self.config.getProperty('LabelUI', 'defaultImage_w', type=int, fallback=800),
            'defaultImage_h': self.config.getProperty('LabelUI', 'defaultImage_h', type=int, fallback=600),
            'styles': styles['styles'],
            'backdrops': backdrops,
            'welcomeMessage': welcomeMessage,
            'demoMode': self.config.getProperty('Project', 'demoMode', type=bool, fallback=False)
        }


    def _assemble_annotations(self, cursor):
        response = {}
        while True:
            b = cursor.fetchone()
            if b is None:
                break

            imgID = str(b['image'])
            if not imgID in response:
                response[imgID] = {
                    'fileName': b['filename'],
                    'predictions': {},
                    'annotations': {}
                }
            viewcount = b['viewcount']
            if viewcount is not None:
                response[imgID]['viewcount'] = viewcount

            # parse annotations and predictions
            entryID = str(b['id'])
            if b['ctype'] is not None:
                colnames = self.sqlBuilder.getColnames(b['ctype'])
                entry = {}
                for c in colnames:
                    value = b[c]
                    if isinstance(value, datetime):
                        value = value.timestamp()
                    elif isinstance(value, UUID):
                        value = str(value)
                    entry[c] = value
                
                if b['ctype'] == 'annotation':
                    response[imgID]['annotations'][entryID] = entry
                elif b['ctype'] == 'prediction':
                    response[imgID]['predictions'][entryID] = entry
        
        return response


    def getProjectSettings(self):
        '''
            Queries the database for general project-specific metadata, such as:
            - Classes: names, indices, default colors
            - Annotation type: one of {class labels, positions, bboxes}
        '''
        return self.projectSettings


    def getProjectInfo(self):
        '''
            Returns safe, shareable information about the project.
        '''
        return {
            'projectName' : self.projectSettings['projectName'],
            'projectDescription' : self.projectSettings['projectDescription'],
            'demoMode': self.config.getProperty('Project', 'demoMode', type=bool, fallback=False),
            'backdrops': self.projectSettings['backdrops']['images']
        }


    def getClassDefinitions(self):
        '''
            Returns a dictionary with entries for all classes in the project.
        '''
        classdef = {
            'entries': {
                'default': {}   # default group for ungrouped label classes
            }
        }
        schema = self.config.getProperty('Database', 'schema')

        # query data
        sql = '''
            SELECT 'group' AS type, id, NULL as idx, name, color, parent FROM {schema}.labelclassgroup
            UNION ALL
            SELECT 'class' AS type, id, idx, name, color, labelclassgroup FROM {schema}.labelclass;
        '''.format(schema=schema)
        classData = self.dbConnector.execute(sql, None, 'all')

        # assemble entries first
        allEntries = {}
        numClasses = 0
        for cl in classData:
            id = str(cl['id'])
            entry = {
                'id': id,
                'name': cl['name'],
                'color': cl['color'],
                'parent': str(cl['parent']) if cl['parent'] is not None else None
            }
            if cl['type'] == 'group':
                entry['entries'] = {}
            else:
                entry['index'] = cl['idx']
                numClasses += 1
            allEntries[id] = entry
        

        # transform into tree
        def _find_parent(tree, parentID):
            if parentID is None:
                return tree['entries']['default']
            elif 'id' in tree and tree['id'] == parentID:
                return tree
            elif 'entries' in tree:
                for ek in tree['entries'].keys():
                    rv = _find_parent(tree['entries'][ek], parentID)
                    if rv is not None:
                        return rv
                return None
            else:
                return None


        allEntries['default'] = {
            'name': '(other)',
            'entries': {}
        }
        allEntries = {
            'entries': allEntries
        }
        for key in list(allEntries['entries'].keys()):
            if key == 'default':
                continue
            if key in allEntries['entries']:
                entry = allEntries['entries'][key]
                parentID = entry['parent']
                del entry['parent']

                if 'entries' in entry and parentID is None:
                    # group, but no parent: append to root directly
                    allEntries['entries'][key] = entry
                
                else:
                    # move item
                    parent = _find_parent(allEntries, parentID)
                    parent['entries'][key] = entry
                    del allEntries['entries'][key]

        classdef = allEntries
        classdef['numClasses'] = numClasses
        return classdef


    def getBatch(self, username, data):
        '''
            Returns entries from the database based on the list of data entry identifiers specified.
        '''
        # query
        sql = self.sqlBuilder.getFixedImagesQueryString(self.projectSettings['demoMode'])

        # parse results
        queryVals = (tuple(UUID(d) for d in data), username, username,)
        if self.projectSettings['demoMode']:
            queryVals = (tuple(UUID(d) for d in data),)

        with self.dbConnector.execute_cursor(sql, queryVals) as cursor:
            try:
                response = self._assemble_annotations(cursor)
                # self.dbConnector.conn.commit()
            except Exception as e:
                print(e)
                # self.dbConnector.conn.rollback()
            finally:
                pass
                # cursor.close()
        return { 'entries': response }
        
    
    def getNextBatch(self, username, order='unlabeled', subset='default', limit=None):
        '''
            TODO: description
        '''
        # query
        sql = self.sqlBuilder.getNextBatchQueryString(order, subset, self.projectSettings['demoMode'])

        # limit (TODO: make 128 a hyperparameter)
        if limit is None:
            limit = 128
        else:
            limit = min(int(limit), 128)

        # parse results
        queryVals = (limit,username,)
        if self.projectSettings['demoMode']:
            queryVals = (limit,)

        with self.dbConnector.execute_cursor(sql, queryVals) as cursor:
            response = self._assemble_annotations(cursor)

        return { 'entries': response }


    def submitAnnotations(self, username, submissions):
        '''
            Sends user-provided annotations to the database.
        '''
        if self.projectSettings['demoMode']:
            return 0

        # assemble values
        colnames = getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value
        values_insert = []
        values_update = []

        meta = (None if not 'meta' in submissions else json.dumps(submissions['meta']))

        # for deletion: remove all annotations whose image ID matches but whose annotation ID is not among the submitted ones
        ids = []

        viewcountValues = []
        for imageKey in submissions['entries']:
            entry = submissions['entries'][imageKey]

            try:
                lastChecked = entry['timeCreated']
                lastTimeRequired = entry['timeRequired']
                if lastTimeRequired is None: lastTimeRequired = 0
            except:
                lastChecked = datetime.now(tz=pytz.utc)
                lastTimeRequired = 0

            if 'annotations' in entry and len(entry['annotations']):
                for annotation in entry['annotations']:
                    # assemble annotation values
                    annotationTokens = self.annoParser.parseAnnotation(annotation)
                    annoValues = []
                    for cname in colnames:
                        if cname == 'id':
                            if cname in annotationTokens:
                                # cast and only append id if the annotation is an existing one
                                annoValues.append(UUID(annotationTokens[cname]))
                                ids.append(UUID(annotationTokens[cname]))
                        elif cname == 'image':
                            annoValues.append(UUID(imageKey))
                        elif cname == 'label' and annotationTokens[cname] is not None:
                            annoValues.append(UUID(annotationTokens[cname]))
                        elif cname == 'timeCreated':
                            try:
                                annoValues.append(dateutil.parser.parse(annotationTokens[cname]))
                            except:
                                annoValues.append(datetime.now(tz=pytz.utc))
                        elif cname == 'timeRequired':
                            timeReq = annotationTokens[cname]
                            if timeReq is None: timeReq = 0
                            annoValues.append(timeReq)
                        elif cname == 'username':
                            annoValues.append(username)
                        elif cname in annotationTokens:
                            annoValues.append(annotationTokens[cname])
                        elif cname == 'unsure':
                            if 'unsure' in annotationTokens and annotationTokens['unsure'] is not None:
                                annoValues.append(annotationTokens[cname])
                            else:
                                annoValues.append(False)
                        elif cname == 'meta':
                            annoValues.append(meta)
                        else:
                            annoValues.append(None)
                    if 'id' in annotationTokens:
                        # existing annotation; update
                        values_update.append(tuple(annoValues))
                    else:
                        # new annotation
                        values_insert.append(tuple(annoValues))
                    
            viewcountValues.append((username, imageKey, 1, lastChecked, lastTimeRequired, meta))


        schema = self.config.getProperty('Database', 'schema')


        # delete all annotations that are not in submitted batch
        imageKeys = list(UUID(k) for k in submissions['entries'])
        if len(ids):
            sql = '''
                DELETE FROM {schema}.annotation WHERE username = %s AND id IN (
                    SELECT idQuery.id FROM (
                        SELECT * FROM {schema}.annotation WHERE id NOT IN %s
                    ) AS idQuery
                    JOIN (
                        SELECT * FROM {schema}.annotation WHERE image IN %s
                    ) AS imageQuery ON idQuery.id = imageQuery.id);
            '''.format(schema=schema)
            self.dbConnector.execute(sql, (username, tuple(ids), tuple(imageKeys),))
        else:
            # no annotations submitted; delete all annotations submitted before
            sql = '''
                DELETE FROM {schema}.annotation WHERE username = %s AND image IN %s;
            '''.format(schema=schema)
            self.dbConnector.execute(sql, (username, tuple(imageKeys),))

        # insert new annotations
        if len(values_insert):
            sql = '''
                INSERT INTO {}.annotation ({})
                VALUES %s ;
            '''.format(
                schema,
                ', '.join(colnames[1:])     # skip 'id' column
            )
            self.dbConnector.insert(sql, values_insert)

        # update existing annotations
        if len(values_update):
            updateCols = ''
            for col in colnames:
                if col == 'label':
                    updateCols += '{col} = UUID(e.{col}),'.format(col=col)
                elif col == 'timeRequired':
                    # we sum the required times together
                    updateCols += '{col} = COALESCE(a.{col},0) + COALESCE(e.{col},0),'.format(col=col)
                else:
                    updateCols += '{col} = e.{col},'.format(col=col)

            sql = '''
                UPDATE {schema}.annotation AS a
                SET {updateCols}
                FROM (VALUES %s) AS e({colnames})
                WHERE e.id = a.id;
            '''.format(
                schema=schema,
                updateCols=updateCols.strip(','),
                colnames=', '.join(colnames)
            )
            self.dbConnector.insert(sql, values_update)


        # viewcount table
        sql = '''
            INSERT INTO {}.image_user (username, image, viewcount, last_checked, last_time_required, meta)
            VALUES %s 
            ON CONFLICT (username, image) DO UPDATE SET viewcount = image_user.viewcount + 1, last_checked = EXCLUDED.last_checked, last_time_required = EXCLUDED.last_time_required, meta = EXCLUDED.meta;
        '''.format(schema)

        self.dbConnector.insert(sql, viewcountValues)


        return 0