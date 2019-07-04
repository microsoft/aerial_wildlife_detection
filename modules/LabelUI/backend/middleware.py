'''
    Definition of the layer between the UI frontend and the database.

    2019 Benjamin Kellenberger
'''

from uuid import UUID
from datetime import datetime
import pytz
import dateutil.parser
from modules.Database.app import Database
from .sql_string_builder import SQLStringBuilder
from .annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction, parseAnnotation


class DBMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self._fetchProjectSettings()
        self.sqlBuilder = SQLStringBuilder(config)
        # self._initSQLstrings()


    def _fetchProjectSettings(self):
        self.projectSettings = {
            'projectName': self.config.getProperty('Project', 'projectName'),
            'projectDescription': self.config.getProperty('Project', 'projectDescription'),
            'dataServerURI': self.config.getProperty('LabelUI', 'dataServer_uri'),
            'dataType': self.config.getProperty('Project', 'dataType'),
            'minObjSize': self.config.getProperty('Project', 'minObjSize'),
            'classes': self.getClassDefinitions(),
            'enableEmptyClass': self.config.getProperty('Project', 'enableEmptyClass'),
            'annotationType': self.config.getProperty('Project', 'annotationType'),
            'predictionType': self.config.getProperty('Project', 'predictionType'),
            'showPredictions': self.config.getProperty('LabelUI', 'showPredictions'),
            'showPredictions_minConf': self.config.getProperty('LabelUI', 'showPredictions_minConf'),
            'carryOverPredictions': self.config.getProperty('LabelUI', 'carryOverPredictions'),
            'carryOverRule': self.config.getProperty('LabelUI', 'carryOverRule'),
            'carryOverPredictions_minConf': self.config.getProperty('LabelUI', 'carryOverPredictions_minConf'),
            'defaultBoxSize_w': self.config.getProperty('LabelUI', 'defaultBoxSize_w'),
            'defaultBoxSize_h': self.config.getProperty('LabelUI', 'defaultBoxSize_h'),
            'numImages_x': self.config.getProperty('LabelUI', 'numImages_x', 3),
            'numImages_y': self.config.getProperty('LabelUI', 'numImages_y', 2),
            'defaultImage_w': self.config.getProperty('LabelUI', 'defaultImage_w', 800),
            'defaultImage_h': self.config.getProperty('LabelUI', 'defaultImage_h', 600),
        }


    def _assemble_annotations(self, cursor, limit=1e9):
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
            'projectDescription' : self.projectSettings['projectDescription']
        }


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
            classProps[c]['id'] = str(classProps[c]['id'])  # convert UUID to string
            classID = classProps[c]['id']
            classdef[classID] = classProps[c]

        return classdef


    def getBatch(self, username, data):
        '''
            Returns entries from the database based on the list of data entry identifiers specified.
        '''
        # query
        sql = self.sqlBuilder.getFixedImagesQueryString()

        # parse results
        with self.dbConnector.execute_cursor(sql, (tuple(UUID(d) for d in data), username, username,)) as cursor:
            try:
                response = self._assemble_annotations(cursor)
                self.dbConnector.conn.commit()
            except:
                self.dbConnector.conn.rollback()
            finally:
                cursor.close()
        return { 'entries': response }
        
    
    def getNextBatch(self, username, order='unlabeled', subset='default', limit=None):
        '''
            TODO: description
        '''
        # query
        sql = self.sqlBuilder.getNextBatchQueryString(order, subset)

        # limit (TODO: make 128 a hyperparameter)
        if limit is None:
            limit = 128
        else:
            limit = min(int(limit), 128)

        # parse results
        with self.dbConnector.execute_cursor(sql, (username,limit,username,)) as cursor:
            try:
                response = self._assemble_annotations(cursor, limit)
                self.dbConnector.conn.commit()
            except:
                self.dbConnector.conn.rollback()
            finally:
                cursor.close()

            # #TODO
            # if len(response) == 1:
            #     import os
            #     from PIL import Image
            #     import matplotlib
            #     matplotlib.use('TkAgg')
            #     import matplotlib.pyplot as plt
            #     from matplotlib.patches import Rectangle
            #     img = Image.open(os.path.join('/datadrive/hfaerialblobs/bkellenb/predictions/A/sde-A_20180921A/images', b['filename']))
            #     sz = img.size
            #     plt.figure(1)
            #     plt.imshow(img)
            #     ax = plt.gca()
            #     for key in response[imgID]['predictions']:
            #         pred = response[imgID]['predictions'][key]
            #         ax.add_patch(Rectangle(
            #             (sz[0] * (pred['x'] - pred['width']/2), sz[1] * (pred['y'] - pred['height']/2),),
            #             sz[0] * pred['width'], sz[1] * pred['height'],
            #             fill=False,
            #             ec='r'
            #         ))
            #     plt.draw()
            #     plt.waitforbuttonpress()

        return { 'entries': response }


    def submitAnnotations(self, username, submissions):
        '''
            Sends user-provided annotations to the database.
        '''
        
        # #TODO
        # schema = self.config.getProperty('Database', 'schema')
        # imageID = list(submissions['entries'].keys())[0]
        # filename = self.dbConnector.execute("SELECT filename FROM {}.image WHERE id = %s".format(schema),(imageID,), numReturn=1)
        # filename = filename[0]['filename']
        # import os
        # from PIL import Image
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # img = Image.open(os.path.join('/datadrive/hfaerialblobs/bkellenb/predictions/A/sde-A_20180921A/images', filename))
        # sz = img.size
        # plt.figure(2)
        # plt.clf()
        # plt.imshow(img)
        # ax = plt.gca()
        # annos = submissions['entries'][imageID]['annotations']
        # for key in annos:
        #     geom = annos[key]['geometry']['coordinates']
        #     ax.add_patch(Rectangle(
        #         (sz[0] * (geom[0] - geom[2]/2), sz[1] * (geom[1] - geom[3]/2),),
        #         sz[0] * geom[2], sz[1] * geom[3],
        #         fill=False,
        #         ec='r'
        #     ))
        # plt.draw()
        # plt.waitforbuttonpress()


        # assemble values
        colnames = getattr(QueryStrings_annotation, self.projectSettings['annotationType']).value[0]
        values_insert = []
        values_update = []

        # for deletion: remove all annotations whose image ID matches but whose annotation ID is not among the submitted ones
        ids = []

        viewcountValues = []
        for imageKey in submissions['entries']:
            entry = submissions['entries'][imageKey]

            try:
                lastChecked = entry['timeCreated']
                lastTimeRequired = entry['timeRequired']
            except:
                #TODO
                lastChecked = datetime.now(tz=pytz.utc)
                lastTimeRequired = 0

            if 'annotations' in entry and len(entry['annotations']):
                for annotation in entry['annotations']:
                    # assemble annotation values
                    annotationTokens = parseAnnotation(annotation)
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
                        elif cname == 'username':
                            annoValues.append(username)
                        elif cname in annotationTokens:
                            annoValues.append(annotationTokens[cname])
                        else:
                            annoValues.append(None)
                    if 'id' in annotationTokens:
                        # existing annotation; update
                        values_update.append(tuple(annoValues))
                    else:
                        # new annotation
                        values_insert.append(tuple(annoValues))
                    
            viewcountValues.append((username, imageKey, 1, lastChecked, lastTimeRequired))


        schema = self.config.getProperty('Database', 'schema')


        # delete all annotations that are not in submitted batch
        if len(ids):
            imageKeys = list(UUID(k) for k in submissions['entries'])
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
            INSERT INTO {}.image_user (username, image, viewcount, last_checked, last_time_required)
            VALUES %s 
            ON CONFLICT (username, image) DO UPDATE SET viewcount = image_user.viewcount + 1, last_checked = EXCLUDED.last_checked, last_time_required = EXCLUDED.last_time_required;
        '''.format(schema)

        self.dbConnector.insert(sql, viewcountValues)


        return 0