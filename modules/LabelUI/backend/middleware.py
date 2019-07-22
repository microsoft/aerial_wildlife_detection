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
from .annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction, AnnotationParser


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

        self.projectSettings = {
            'projectName': self.config.getProperty('Project', 'projectName'),
            'projectDescription': self.config.getProperty('Project', 'projectDescription'),
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
            'numImages_x': self.config.getProperty('LabelUI', 'numImages_x', type=int, fallback=3),
            'numImages_y': self.config.getProperty('LabelUI', 'numImages_y', type=int, fallback=2),
            'defaultImage_w': self.config.getProperty('LabelUI', 'defaultImage_w', type=int, fallback=800),
            'defaultImage_h': self.config.getProperty('LabelUI', 'defaultImage_h', type=int, fallback=600),
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
                # self.dbConnector.conn.commit()
            except:
                pass
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
        sql = self.sqlBuilder.getNextBatchQueryString(order, subset)

        # limit (TODO: make 128 a hyperparameter)
        if limit is None:
            limit = 128
        else:
            limit = min(int(limit), 128)

        # parse results
        with self.dbConnector.execute_cursor(sql, (username,limit,username,)) as cursor:
            response = self._assemble_annotations(cursor)

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