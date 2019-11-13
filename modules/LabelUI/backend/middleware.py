'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

import os
import ast
from uuid import UUID
from datetime import datetime
import pytz
import dateutil.parser
import json
from psycopg2 import sql
from modules.Database.app import Database
from .sql_string_builder import SQLStringBuilder
from .annotation_sql_tokens import QueryStrings_annotation, AnnotationParser


class DBMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self.project_immutables = {}       # project settings that cannot be changed (project shorthand -> {settings})

        self._fetchProjectSettings()
        self.sqlBuilder = SQLStringBuilder(config)
        self.annoParser = AnnotationParser(config)


    def _fetchProjectSettings(self):
        # AI controller URI
        aiControllerURI = self.config.getProperty('Server', 'aiController_uri')
        if aiControllerURI is None or aiControllerURI.strip() == '':
            # no AI backend configured
            aiControllerURI = None

        # # LabelUI drawing styles
        # with open(self.config.getProperty('LabelUI', 'styles_file', type=str, fallback='modules/LabelUI/static/json/styles.json'), 'r') as f:
        #     styles = json.load(f)

        # # Image backdrops for index screen
        # with open(self.config.getProperty('Project', 'backdrops_file', type=str, fallback='modules/LabelUI/static/json/backdrops.json'), 'r') as f:
        #     backdrops = json.load(f)

        # # Welcome message for UI tutorial
        # with open(self.config.getProperty('Project', 'welcome_message_file', type=str, fallback='modules/LabelUI/static/templates/welcome_message.html'), 'r') as f:
        #     welcomeMessage = f.readlines()

        # global, project-independent settings
        self.globalSettings = {
            'indexURI': self.config.getProperty('Server', 'index_uri', type=str, fallback='/'),
            'dataServerURI': self.config.getProperty('Server', 'dataServer_uri'),
            'aiControllerURI': aiControllerURI,
            'dataType': self.config.getProperty('Project', 'dataType', fallback='images'),      #TODO
        }
        
        # self.projectSettings = {
        #     # 'projectName': self.config.getProperty('Project', 'projectName'),
        #     # 'projectDescription': self.config.getProperty('Project', 'projectDescription'),
        #     # 'indexURI': self.config.getProperty('Server', 'index_uri', type=str, fallback='/'),
        #     # 'dataServerURI': self.config.getProperty('Server', 'dataServer_uri'),
        #     # 'aiControllerURI': aiControllerURI,
        #     # 'dataType': self.config.getProperty('Project', 'dataType', fallback='images'),
        #     # 'classes': self.getClassDefinitions(),
        #     # 'enableEmptyClass': self.config.getProperty('Project', 'enableEmptyClass', fallback='no'),
        #     # 'annotationType': self.config.getProperty('Project', 'annotationType'),
        #     # 'predictionType': self.config.getProperty('Project', 'predictionType'),
        #     'showPredictions': self.config.getProperty('LabelUI', 'showPredictions', fallback='yes'),
        #     'showPredictions_minConf': self.config.getProperty('LabelUI', 'showPredictions_minConf', type=float, fallback=0.5),
        #     'carryOverPredictions': self.config.getProperty('LabelUI', 'carryOverPredictions', fallback='no'),
        #     'carryOverRule': self.config.getProperty('LabelUI', 'carryOverRule', fallback='maxConfidence'),
        #     'carryOverPredictions_minConf': self.config.getProperty('LabelUI', 'carryOverPredictions_minConf', type=float, fallback=0.75),
        #     'defaultBoxSize_w': self.config.getProperty('LabelUI', 'defaultBoxSize_w', type=int, fallback=10),
        #     'defaultBoxSize_h': self.config.getProperty('LabelUI', 'defaultBoxSize_h', type=int, fallback=10),
        #     'minBoxSize_w': self.config.getProperty('Project', 'box_minWidth', type=int, fallback=1),
        #     'minBoxSize_h': self.config.getProperty('Project', 'box_minHeight', type=int, fallback=1),
        #     'numImagesPerBatch': self.config.getProperty('LabelUI', 'numImagesPerBatch', type=int, fallback=1),
        #     'minImageWidth': self.config.getProperty('LabelUI', 'minImageWidth', type=int, fallback=300),
        #     'numImageColumns_max': self.config.getProperty('LabelUI', 'numImageColumns_max', type=int, fallback=1),
        #     'defaultImage_w': self.config.getProperty('LabelUI', 'defaultImage_w', type=int, fallback=800),
        #     'defaultImage_h': self.config.getProperty('LabelUI', 'defaultImage_h', type=int, fallback=600),
        #     'styles': styles['styles'],
        #     'backdrops': backdrops,
        #     'welcomeMessage': welcomeMessage,
        #     'demoMode': self.config.getProperty('Project', 'demoMode', type=bool, fallback=False)
        # }


    def _assemble_annotations(self, project, cursor):
        response = {}
        while True:
            b = cursor.fetchone()
            if b is None:
                break

            imgID = str(b['image'])
            if not imgID in response:
                response[imgID] = {
                    'fileName': os.path.join(project, b['filename']),
                    'predictions': {},
                    'annotations': {},
                    'last_checked': None
                }
            viewcount = b['viewcount']
            if viewcount is not None:
                response[imgID]['viewcount'] = viewcount
            last_checked = b['last_checked']
            if last_checked is not None:
                if response[imgID]['last_checked'] is None:
                    response[imgID]['last_checked'] = last_checked
                else:
                    response[imgID]['last_checked'] = max(response[imgID]['last_checked'], last_checked)

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


    def get_project_immutables(self, project):
        if project not in self.project_immutables:
            queryStr = 'SELECT annotationType, predictionType, demoMode FROM aide_admin.project WHERE shortname = %s;'
            result = self.dbConnector.execute(queryStr, (project,), 1)
            if result and len(result):
                self.project_immutables[project] = {
                    'annotationType': result[0]['annotationtype'],
                    'predictionType': result[0]['predictiontype'],
                    'demoMode': result[0]['demomode']
                }
            else:
                return None
        return self.project_immutables[project]

    
    def get_dynamic_project_settings(self, project):
        queryStr = 'SELECT ui_settings FROM aide_admin.project WHERE shortname = %s;'
        result = self.dbConnector.execute(queryStr, (project,), 1)
        return ast.literal_eval(result[0]['ui_settings'])


    def getProjectSettings(self, project):
        '''
            Queries the database for general project-specific metadata, such as:
            - Classes: names, indices, default colors
            - Annotation type: one of {class labels, positions, bboxes}
        '''
        # publicly available info from DB
        projSettings = self.getProjectInfo(project)

        # label classes
        projSettings['classes'] = self.getClassDefinitions(project)

        # static and dynamic project settings and properties from configuration file
        projSettings = { **projSettings, **self.get_project_immutables(project), **self.get_dynamic_project_settings(project), **self.globalSettings }

        # append project shorthand to AIController URI 
        if 'aiControllerURI' in projSettings and projSettings['aiControllerURI'] is not None and len(projSettings['aiControllerURI']):
            projSettings['aiControllerURI'] = os.path.join(projSettings['aiControllerURI'], project) + '/'

        return projSettings


    def getProjectInfo(self, project):
        '''
            Returns safe, shareable information about the project
            (i.e., users don't need to be part of the project to see these data).
        '''
        queryStr = '''
            SELECT shortname, name, description, demoMode,
            interface_enabled
            FROM aide_admin.project
            WHERE shortname = %s
        '''
        result = self.dbConnector.execute(queryStr, (project,), 1)[0]

        # # TODO: append backdrops
        # result['backdrops'] = self.projectSettings['backdrops']['images']

        return {
            'projectShortname': result['shortname'],
            'projectName': result['name'],
            'projectDescription': result['description'],
            'demoMode': result['demomode'],
            'interfaceEnabled': result['interface_enabled']
        }


    def getClassDefinitions(self, project):
        '''
            Returns a dictionary with entries for all classes in the project.
        '''
        classdef = {
            'entries': {
                'default': {}   # default group for ungrouped label classes
            }
        }

        # query data
        queryStr = sql.SQL('''
            SELECT 'group' AS type, id, NULL as idx, name, color, parent, NULL AS keystroke FROM {}
            UNION ALL
            SELECT 'class' AS type, id, idx, name, color, labelclassgroup, keystroke FROM {};
            ''').format(
                sql.Identifier(project, 'labelclassgroup'),
                sql.Identifier(project, 'labelclass')
            )

        # queryStr = '''
        #     SELECT 'group' AS type, id, NULL as idx, name, color, parent, NULL AS keystroke FROM {schema}.labelclassgroup
        #     UNION ALL
        #     SELECT 'class' AS type, id, idx, name, color, labelclassgroup, keystroke FROM {schema}.labelclass;
        # '''.format(schema=schema)
        classData = self.dbConnector.execute(queryStr, None, 'all')

        # assemble entries first
        allEntries = {}
        numClasses = 0
        for cl in classData:
            id = str(cl['id'])
            entry = {
                'id': id,
                'name': cl['name'],
                'color': cl['color'],
                'parent': str(cl['parent']) if cl['parent'] is not None else None,
            }
            if cl['type'] == 'group':
                entry['entries'] = {}
            else:
                entry['index'] = cl['idx']
                entry['keystroke'] = cl['keystroke']
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


    def getBatch_fixed(self, project, username, data):
        '''
            Returns entries from the database based on the list of data entry identifiers specified.
        '''
        # query
        projImmutables = self.get_project_immutables(project)
        queryStr = self.sqlBuilder.getFixedImagesQueryString(project, projImmutables['annotationType'], projImmutables['predictionType'], projImmutables['demoMode'])

        # parse results
        queryVals = (tuple(UUID(d) for d in data), username, username,)
        if projImmutables['demoMode']:
            queryVals = (tuple(UUID(d) for d in data),)

        with self.dbConnector.execute_cursor(queryStr, queryVals) as cursor:
            try:
                response = self._assemble_annotations(project, cursor)
                # self.dbConnector.conn.commit()
            except Exception as e:
                print(e)
                # self.dbConnector.conn.rollback()
            finally:
                pass
                # cursor.close()
        return { 'entries': response }
        

    def getBatch_auto(self, project, username, order='unlabeled', subset='default', limit=None):
        '''
            TODO: description
        '''
        # query
        projImmutables = self.get_project_immutables(project)
        queryStr = self.sqlBuilder.getNextBatchQueryString(project, projImmutables['annotationType'], projImmutables['predictionType'], order, subset, projImmutables['demoMode'])

        # limit (TODO: make 128 a hyperparameter)
        if limit is None:
            limit = 128
        else:
            limit = min(int(limit), 128)

        # parse results
        queryVals = (username,limit,username,)
        if projImmutables['demoMode']:
            queryVals = (limit,)

        with self.dbConnector.execute_cursor(queryStr, queryVals) as cursor:
            response = self._assemble_annotations(project, cursor)

        return { 'entries': response }


    def getBatch_timeRange(self, project, minTimestamp, maxTimestamp, userList, skipEmptyImages=False, limit=None):
        '''
            Returns images that have been annotated within the given time range and/or
            by the given user(s). All arguments are optional.
            Useful for reviewing existing annotations.
        '''
        # query string
        projImmutables = self.get_project_immutables(project)
        queryStr = self.sqlBuilder.getDateQueryString(project, projImmutables['annotationType'], minTimestamp, maxTimestamp, userList, skipEmptyImages)

        # check validity and provide arguments
        queryVals = []
        if userList is not None:
            queryVals.append(tuple(userList))
        if minTimestamp is not None:
            queryVals.append(minTimestamp)
        if maxTimestamp is not None:
            queryVals.append(maxTimestamp)
        if skipEmptyImages and userList is not None:
            queryVals.append(tuple(userList))

        # limit (TODO: make 128 a hyperparameter)
        if limit is None:
            limit = 128
        else:
            limit = min(int(limit), 128)
        queryVals.append(limit)

        if userList is not None:
            queryVals.append(tuple(userList))

        # query and parse results
        with self.dbConnector.execute_cursor(queryStr, tuple(queryVals)) as cursor:
            try:
                response = self._assemble_annotations(project, cursor)
                # self.dbConnector.conn.commit()
            except Exception as e:
                print(e)
                # self.dbConnector.conn.rollback()
            finally:
                pass
                # cursor.close()
        return { 'entries': response }

    
    def get_timeRange(self, project, userList, skipEmptyImages=False):
        '''
            Returns two timestamps denoting the temporal limits within which
            images have been viewed by the users provided in the userList.
            Arguments:
            - userList: string (single user name) or list of strings (multiple).
                        Can also be None; in this case all annotations will be
                        checked.
            - skipEmptyImages: if True, only images that contain at least one
                               annotation will be considered.
        '''
        # query string
        queryStr = self.sqlBuilder.getTimeRangeQueryString(project, userList, skipEmptyImages)

        arguments = (None if userList is None else tuple(userList))
        result = self.dbConnector.execute(queryStr, (arguments,), numReturn=1)

        if result is not None and len(result):
            return {
                'minTimestamp': result[0]['mintimestamp'],
                'maxTimestamp': result[0]['maxtimestamp'],
            }
        else:
            return {
                'error': 'no annotations made'
            }


    def submitAnnotations(self, project, username, submissions):
        '''
            Sends user-provided annotations to the database.
        '''

        projImmutables = self.get_project_immutables(project)
        if projImmutables['demoMode']:
            return 0

        # assemble values
        colnames = getattr(QueryStrings_annotation, projImmutables['annotationType']).value
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


        # delete all annotations that are not in submitted batch
        imageKeys = list(UUID(k) for k in submissions['entries'])
        if len(imageKeys):
            if len(ids):
                queryStr = sql.SQL('''
                    DELETE FROM {id_anno} WHERE username = %s AND id IN (
                        SELECT idQuery.id FROM (
                            SELECT * FROM {id_anno} WHERE id NOT IN %s
                        ) AS idQuery
                        JOIN (
                            SELECT * FROM {id_anno} WHERE image IN %s
                        ) AS imageQuery ON idQuery.id = imageQuery.id);
                ''').format(
                    id_anno=sql.Identifier(project, 'annotation'))
                self.dbConnector.execute(queryStr, (username, tuple(ids), tuple(imageKeys),))
            else:
                # no annotations submitted; delete all annotations submitted before
                queryStr = sql.SQL('''
                    DELETE FROM {id_anno} WHERE username = %s AND image IN %s;
                ''').format(
                    id_anno=sql.Identifier(project, 'annotation'))
                self.dbConnector.execute(queryStr, (username, tuple(imageKeys),))

        # insert new annotations
        if len(values_insert):
            queryStr = sql.SQL('''
                INSERT INTO {id_anno} ({cols})
                VALUES %s ;
            ''').format(
                id_anno=sql.Identifier(project, 'annotation'),
                cols=sql.SQL(', ').join([sql.SQL(c) for c in colnames[1:]])     # skip 'id' column
            )
            self.dbConnector.insert(queryStr, values_insert)

        # update existing annotations
        if len(values_update):

            updateCols = []
            for col in colnames:
                if col == 'label':
                    updateCols.append(sql.SQL('label = UUID(e.label)'))
                elif col == 'timeRequired':
                    # we sum the required times together
                    updateCols.append(sql.SQL('timeRequired = COALESCE(a.timeRequired,0) + COALESCE(e.timeRequired,0)'))
                else:
                    updateCols.append(sql.SQL('{col} = e.{col}').format(col=sql.SQL(col)))

            queryStr = sql.SQL('''
                UPDATE {id_anno} AS a
                SET {updateCols}
                FROM (VALUES %s) AS e({colnames})
                WHERE e.id = a.id
            ''').format(
                id_anno=sql.Identifier(project, 'annotation'),
                updateCols=sql.SQL(', ').join(updateCols),
                colnames=sql.SQL(', ').join([sql.SQL(c) for c in colnames])
            )

            # #TODO: deprecated:
            # updateCols = ''
            # for col in colnames:
            #     if col == 'label':
            #         updateCols += '{col} = UUID(e.{col}),'.format(col=col)
            #     elif col == 'timeRequired':
            #         # we sum the required times together
            #         updateCols += '{col} = COALESCE(a.{col},0) + COALESCE(e.{col},0),'.format(col=col)
            #     else:
            #         updateCols += '{col} = e.{col},'.format(col=col)

            # sql = '''
            #     UPDATE {schema}.annotation AS a
            #     SET {updateCols}
            #     FROM (VALUES %s) AS e({colnames})
            #     WHERE e.id = a.id;
            # '''.format(
            #     schema=schema,
            #     updateCols=updateCols.strip(','),
            #     colnames=', '.join(colnames)
            # )
            self.dbConnector.insert(queryStr, values_update)


        # viewcount table
        queryStr = sql.SQL('''
            INSERT INTO {id_iu} (username, image, viewcount, last_checked, last_time_required, meta)
            VALUES %s 
            ON CONFLICT (username, image) DO UPDATE SET viewcount = image_user.viewcount + 1, last_checked = EXCLUDED.last_checked, last_time_required = EXCLUDED.last_time_required, meta = EXCLUDED.meta;
        ''').format(
            id_iu=sql.Identifier(project, 'image_user')
        )
        self.dbConnector.insert(queryStr, viewcountValues)


        return 0