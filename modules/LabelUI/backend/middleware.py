'''
    Definition of the layer between the UI frontend and the database.

    2019-20 Benjamin Kellenberger
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
        self.sqlBuilder = SQLStringBuilder()
        self.annoParser = AnnotationParser(config)


    def _fetchProjectSettings(self):
        # AI controller URI
        aiControllerURI = self.config.getProperty('Server', 'aiController_uri')
        if aiControllerURI is None or aiControllerURI.strip() == '':
            # no AI backend configured
            aiControllerURI = None

        # global, project-independent settings
        self.globalSettings = {
            'indexURI': self.config.getProperty('Server', 'index_uri', type=str, fallback='/'),
            'dataServerURI': self.config.getProperty('Server', 'dataServer_uri'),
            'aiControllerURI': aiControllerURI,
            'dataType': self.config.getProperty('Project', 'dataType', fallback='images'),      #TODO
        }


    def _assemble_annotations(self, project, cursor, hideGoldenQuestionInfo):
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

            if not hideGoldenQuestionInfo:
                response[imgID]['isGoldenQuestion'] = b['isgoldenquestion']

            # parse annotations and predictions
            entryID = str(b['id'])
            if b['ctype'] is not None:
                colnames = self.sqlBuilder.getColnames(
                    self.project_immutables[project]['annotationType'],
                    self.project_immutables[project]['predictionType'],
                    b['ctype'])
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


    def _set_images_requested(self, project, imageIDs):
        '''
            Sets column "last_requested" of relation "image"
            to the current date. This is done during image
            querying to signal that an image has been requested,
            but not (yet) viewed.
        '''
        # prepare insertion values
        now = datetime.now(tz=pytz.utc)
        vals = []
        for key in imageIDs:
            vals.append(key)
        queryStr = sql.SQL('''
            UPDATE {id_img}
            SET last_requested = %s
            WHERE id IN %s;
        ''').format(id_img=sql.Identifier(project, 'image'))
        self.dbConnector.execute(queryStr, (now, tuple(vals),), None)


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
        return json.loads(result[0]['ui_settings'])       #TODO: ast.literal_eval(result[0]['ui_settings'])


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
            interface_enabled, ai_model_enabled,
            ai_model_library, ai_alcriterion_library
            FROM aide_admin.project
            WHERE shortname = %s
        '''
        result = self.dbConnector.execute(queryStr, (project,), 1)[0]

        # provide flag if AI model is available
        aiModelAvailable = all([
            result['ai_model_enabled'],
            result['ai_model_library'] is not None and len(result['ai_model_library']),
            result['ai_alcriterion_library'] is not None and len(result['ai_alcriterion_library'])
        ])

        return {
            'projectShortname': result['shortname'],
            'projectName': result['name'],
            'projectDescription': result['description'],
            'demoMode': result['demomode'],
            'interfaceEnabled': result['interface_enabled'],
            'ai_model_available': aiModelAvailable
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


    def getBatch_fixed(self, project, username, data, hideGoldenQuestionInfo=True):
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
                response = self._assemble_annotations(project, cursor, hideGoldenQuestionInfo)
                # self.dbConnector.conn.commit()
            except Exception as e:
                print(e)
                # self.dbConnector.conn.rollback()
            finally:
                pass
                # cursor.close()

        # mark images as requested
        self._set_images_requested(project, response)

        return { 'entries': response }
        

    def getBatch_auto(self, project, username, order='unlabeled', subset='default', limit=None, hideGoldenQuestionInfo=True):
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
        queryVals = (username,username,limit,username,)
        if projImmutables['demoMode']:
            queryVals = (limit,)

        with self.dbConnector.execute_cursor(queryStr, queryVals) as cursor:
            response = self._assemble_annotations(project, cursor, hideGoldenQuestionInfo)

        # mark images as requested
        self._set_images_requested(project, response)

        return { 'entries': response }


    def getBatch_timeRange(self, project, minTimestamp, maxTimestamp, userList, skipEmptyImages=False, limit=None, goldenQuestionsOnly=False, hideGoldenQuestionInfo=True):
        '''
            Returns images that have been annotated within the given time range and/or
            by the given user(s). All arguments are optional.
            Useful for reviewing existing annotations.
        '''
        # query string
        projImmutables = self.get_project_immutables(project)
        queryStr = self.sqlBuilder.getDateQueryString(project, projImmutables['annotationType'], minTimestamp, maxTimestamp, userList, skipEmptyImages, goldenQuestionsOnly)

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
                response = self._assemble_annotations(project, cursor, hideGoldenQuestionInfo)
                # self.dbConnector.conn.commit()
            except Exception as e:
                print(e)
                # self.dbConnector.conn.rollback()
            finally:
                pass
                # cursor.close()

        # # mark images as requested
        # self._set_images_requested(project, response)


        return { 'entries': response }

    
    def get_timeRange(self, project, userList, skipEmptyImages=False, goldenQuestionsOnly=False):
        '''
            Returns two timestamps denoting the temporal limits within which
            images have been viewed by the users provided in the userList.
            Arguments:
            - userList: string (single user name) or list of strings (multiple).
                        Can also be None; in this case all annotations will be
                        checked.
            - skipEmptyImages: if True, only images that contain at least one
                               annotation will be considered.
            - goldenQuestionsOnly: if True, only images flagged as golden questions
                                   will be shown.
        '''
        # query string
        queryStr = self.sqlBuilder.getTimeRangeQueryString(project, userList, skipEmptyImages, goldenQuestionsOnly)

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
            return 1

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


    def setGoldenQuestions(self, project, submissions):
        '''
            Receives an iterable of tuples (uuid, bool) and updates the
            property "isGoldenQuestion" of the images accordingly.
        '''
        projImmutables = self.get_project_immutables(project)
        if projImmutables['demoMode']:
            return 1
        
        queryStr = sql.SQL('''
            UPDATE {id_img} AS img SET isGoldenQuestion = c.isGoldenQuestion
            FROM (VALUES %s)
            AS c (id, isGoldenQuestion)
            WHERE c.id = img.id;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        self.dbConnector.insert(queryStr, submissions)

        return 0