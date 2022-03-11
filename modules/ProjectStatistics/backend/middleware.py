'''
    Middleware layer for project statistics calculations.

    2019-22 Benjamin Kellenberger
'''

import copy
import uuid
from datetime import datetime
from collections import defaultdict
from psycopg2 import sql
import numpy as np
from .statisticalFormulas import StatisticalFormulas_user, StatisticalFormulas_model
from modules.Database.app import Database
from modules.LabelUI.backend.annotation_sql_tokens import QueryStrings_annotation, AnnotationTypeTokens
from . import accuracy
from util.helpers import base64ToImage


class ProjectStatisticsMiddleware:

    def __init__(self, config, dbConnector):
        self.config = config
        self.dbConnector = dbConnector

        self.annotationTypes = {}       # for quizgame: dict of annotation types per project
    

    def getProjectStatistics(self, project):
        '''
            Returns statistics, such as number of images (seen),
            number of annotations, etc., on a global and per-user,
            but class-agnostic basis.
        '''
        queryStr = sql.SQL('''
            SELECT NULL AS username, COUNT(*) AS num_img, NULL::bigint AS num_anno FROM {id_img}
            UNION ALL
            SELECT NULL AS username, COUNT(DISTINCT(image)) AS num_img, NULL AS num_anno FROM {id_iu}
            UNION ALL
            SELECT NULL AS username, COUNT(DISTINCT(gq.id)) AS num_img, NULL AS num_anno FROM (
                SELECT id FROM {id_img} WHERE isGoldenQuestion = TRUE
            ) AS gq
            UNION ALL
            SELECT NULL AS username, NULL AS num_img, COUNT(DISTINCT(image)) AS num_anno FROM {id_anno}
            UNION ALL
            SELECT NULL AS username, NULL AS num_img, COUNT(*) AS num_anno FROM {id_anno}
            UNION ALL
            SELECT username, iu_cnt AS num_img, anno_cnt AS num_anno FROM (
            SELECT u.username, iu_cnt, anno_cnt
            FROM (
                SELECT DISTINCT(username) FROM (
                    SELECT username FROM {id_auth}
                    WHERE project = %s
                    UNION ALL
                    SELECT username FROM {id_iu}
                    UNION ALL
                    SELECT username FROM {id_anno}
                ) AS uQuery
            ) AS u
            LEFT OUTER JOIN (
                SELECT username, COUNT(*) AS iu_cnt
                FROM {id_iu}
                GROUP BY username
            ) AS iu
            ON u.username = iu.username
            LEFT OUTER JOIN (
                SELECT username, COUNT(*) AS anno_cnt
                FROM {id_anno}
                GROUP BY username
            ) AS anno
            ON u.username = anno.username
            ORDER BY u.username
        ) AS q;
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user'),
            id_anno=sql.Identifier(project, 'annotation'),
            id_auth=sql.Identifier('aide_admin', 'authentication')
        )
        result = self.dbConnector.execute(queryStr, (project,), 'all')

        response = {
            'num_images': result[0]['num_img'],
            'num_viewed': result[1]['num_img'],
            'num_goldenQuestions': result[2]['num_img'],
            'num_annotated': result[3]['num_anno'],
            'num_annotations': result[4]['num_anno']
        }
        if len(result) > 5:
            response['user_stats'] = {}
            for i in range(5, len(result)):
                uStats = {
                    'num_viewed': result[i]['num_img'],
                    'num_annotations': result[i]['num_anno']
                }
                response['user_stats'][result[i]['username']] = uStats
        return response


    def getLabelclassStatistics(self, project):
        '''
            Returns annotation statistics on a per-label class
            basis.
            TODO: does not work for segmentationMasks (no label fields)
        '''
        queryStr = sql.SQL('''
            SELECT lc.name, COALESCE(num_anno, 0) AS num_anno, COALESCE(num_pred, 0) AS num_pred
            FROM {id_lc} AS lc
            FULL OUTER JOIN (
                SELECT label, COUNT(*) AS num_anno
                FROM {id_anno} AS anno
                GROUP BY label
            ) AS annoCnt
            ON lc.id = annoCnt.label
            FULL OUTER JOIN (
                SELECT label, COUNT(*) AS num_pred
                FROM {id_pred} AS pred
                GROUP BY label
            ) AS predCnt
            ON lc.id = predCnt.label
        ''').format(
            id_lc=sql.Identifier(project, 'labelclass'),
            id_anno=sql.Identifier(project, 'annotation'),
            id_pred=sql.Identifier(project, 'prediction')
        )
        result = self.dbConnector.execute(queryStr, None, 'all')

        response = {}
        if result is not None and len(result):
            for i in range(len(result)):
                nextResult = result[i]
                response[nextResult['name']] = {
                    'num_anno': nextResult['num_anno'],
                    'num_pred': nextResult['num_pred']
                }
        return response


    @staticmethod
    def _calc_geometric_stats(tp, fp, fn):
        tp, fp, fn = float(tp), float(fp), float(fn)
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0.0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0.0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0.0
        return precision, recall, f1


    def getPerformanceStatistics(self, project, entities_eval, entity_target, entityType='user', threshold=0.5, goldenQuestionsOnly=True):
        '''
            Compares the accuracy of a list of users or model states with a target
            user.
            The following measures of accuracy are reported, depending on the
            annotation type:
            - image labels: overall accuracy
            - points:
                    RMSE (distance to closest point with the same label; in pixels)
                    overall accuracy (labels)
            - bounding boxes:
                    IoU (max. with any target bounding box, regardless of label)
                    overall accuracy (labels)
            - segmentation masks:
                    TODO

            Value 'threshold' determines the geometric requirement for an annotation to be
            counted as correct (or incorrect) as follows:
                - points: maximum euclidean distance in pixels to closest target
                - bounding boxes: minimum IoU with best matching target

            If 'goldenQuestionsOnly' is True, only images with flag 'isGoldenQuestion' = True
            will be considered for evaluation.
        '''
        entityType = entityType.lower()

        # get annotation and prediction types for project
        annoTypes = self.dbConnector.execute('''SELECT annotationType, predictionType
            FROM aide_admin.project WHERE shortname = %s;''',
            (project,),
            1)
        annoType = annoTypes[0]['annotationtype']
        predType = annoTypes[0]['predictiontype']

        if entityType != 'user' and annoType != predType:
            # different combinations of annotation and prediction types are currently not supported
            raise Exception('Statistics for unequal annotation and AI model prediction types are currently not supported.')

        # for segmentation masks: get label classes and their ordinals      #TODO: implement per-class statistics for all types
        labelClasses = {}
        lcDef = self.dbConnector.execute(sql.SQL('''
            SELECT id, name, idx, color FROM {id_lc};
        ''').format(id_lc=sql.Identifier(project, 'labelclass')),
        None, 'all')
        if lcDef is not None:
            for l in lcDef:
                labelClasses[str(l['id'])] = (l['idx'], l['name'], l['color'])

        else:
            # no label classes defined
            return {}               


        # compose args list and complete query
        queryArgs = [entity_target, tuple(entities_eval)]
        if annoType == 'points' or annoType == 'boundingBoxes':
            queryArgs.append(threshold)
            if annoType == 'points':
                queryArgs.append(threshold)

        if goldenQuestionsOnly:
            sql_goldenQuestion = sql.SQL('''JOIN (
                    SELECT id
                    FROM {id_img}
                    WHERE isGoldenQuestion = true
                ) AS qi
                ON qi.id = q2.image''').format(
                id_img=sql.Identifier(project, 'image')
            )
        else:
            sql_goldenQuestion = sql.SQL('')


        # result tokens
        tokens = {}
        tokens_normalize = []
        if annoType == 'labels':
            tokens = {
                'num_matches': 0,
                'correct': 0,
                'incorrect': 0,
                'overall_accuracy': 0.0
            }
            tokens_normalize = ['overall_accuracy']
        elif annoType == 'points':
            tokens = {
                'num_pred': 0,
                'num_target': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'avg_dist': 0.0
            }
            tokens_normalize = ['avg_dist']
        elif annoType == 'boundingBoxes':
            tokens = {
                'num_pred': 0,
                'num_target': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'avg_iou': 0.0
            }
            tokens_normalize = ['avg_iou']
        elif annoType == 'segmentationMasks':
            tokens = {
                'num_matches': 0,
                'overall_accuracy': 0.0,
                'per_class': {}
            }
            for clID in labelClasses.keys():
                tokens['per_class'][clID] = {
                    'num_matches': 0,
                    'prec': 0.0,
                    'rec': 0.0,
                    'f1': 0.0
                }
            tokens_normalize = []
        
        if entityType == 'user':
            queryStr = getattr(StatisticalFormulas_user, annoType).value
            queryStr = sql.SQL(queryStr).format(
                id_anno=sql.Identifier(project, 'annotation'),
                id_iu=sql.Identifier(project, 'image_user'),
                sql_goldenQuestion=sql_goldenQuestion
            )

        else:
            queryStr = getattr(StatisticalFormulas_model, annoType).value
            queryStr = sql.SQL(queryStr).format(
                id_anno=sql.Identifier(project, 'annotation'),
                id_iu=sql.Identifier(project, 'image_user'),
                id_pred=sql.Identifier(project, 'prediction'),
                sql_goldenQuestion=sql_goldenQuestion
            )

        #TODO: update points query (according to bboxes); re-write stats parsing below

        # get stats
        response = {}
        result = self.dbConnector.execute(queryStr, tuple(queryArgs), 'all')
        if result is not None and len(result):
            for b in result:
                if entityType == 'user':
                    entity = b['username']
                else:
                    entity = str(b['cnnstate'])

                if not entity in response:
                    response[entity] = copy.deepcopy(tokens)
                if annoType in ('points', 'boundingBoxes'):
                    response[entity]['num_matches'] = 1
                    if b['num_target'] > 0:
                        response[entity]['num_matches'] += 1
                
                if annoType == 'segmentationMasks':
                    # decode segmentation masks
                    try:
                        mask_target = np.array(base64ToImage(b['q1segmask'], b['q1width'], b['q1height']))
                        mask_source = np.array(base64ToImage(b['q2segmask'], b['q2width'], b['q2height']))
                        
                        if mask_target.shape == mask_source.shape and np.any(mask_target) and np.any(mask_source):

                            # calculate OA
                            intersection = (mask_target>0) * (mask_source>0)
                            if np.any(intersection):
                                oa = np.mean(mask_target[intersection] == mask_source[intersection])
                                response[entity]['overall_accuracy'] += oa
                                response[entity]['num_matches'] += 1

                            # calculate per-class precision and recall values
                            for clID in labelClasses.keys():
                                idx = labelClasses[clID][0]
                                tp = np.sum((mask_target==idx) * (mask_source==idx))
                                fp = np.sum((mask_target!=idx) * (mask_source==idx))
                                fn = np.sum((mask_target==idx) * (mask_source!=idx))
                                if (tp+fp+fn) > 0:
                                    prec, rec, f1 = self._calc_geometric_stats(tp, fp, fn)
                                    response[entity]['per_class'][clID]['num_matches'] += 1
                                    response[entity]['per_class'][clID]['prec'] += prec
                                    response[entity]['per_class'][clID]['rec'] += rec
                                    response[entity]['per_class'][clID]['f1'] += f1

                    except Exception as e:
                        print(f'TODO: error in segmentation mask statistics calculation ("{str(e)}").')

                else:
                    for key in tokens.keys():
                        if key == 'correct' or key == 'incorrect':
                            # classification
                            correct = b['label_correct']
                            # ignore None
                            if correct is True:
                                response[entity]['correct'] += 1
                                response[entity]['num_matches'] += 1
                            elif correct is False:
                                response[entity]['incorrect'] += 1
                                response[entity]['num_matches'] += 1
                        elif key in b and b[key] is not None:
                            response[entity][key] += b[key]

        for entity in response.keys():
            for t in tokens_normalize:
                if t in response[entity]:
                    if t == 'overall_accuracy':
                        response[entity][t] = float(response[entity]['correct']) / \
                            float(response[entity]['correct'] + response[entity]['incorrect'])
                    elif annoType in ('points', 'boundingBoxes'):
                        response[entity][t] /= response[entity]['num_matches']

            if annoType == 'points' or annoType == 'boundingBoxes':
                prec, rec, f1 = self._calc_geometric_stats(
                    response[entity]['tp'],
                    response[entity]['fp'],
                    response[entity]['fn']
                )
                response[entity]['prec'] = prec
                response[entity]['rec'] = rec
                response[entity]['f1'] = f1

            elif annoType == 'segmentationMasks':
                # normalize OA
                response[entity]['overall_accuracy'] /= response[entity]['num_matches']

                # normalize all label class values as well
                for lcID in labelClasses.keys():
                    numMatches = response[entity]['per_class'][lcID]['num_matches']
                    if numMatches > 0:
                        response[entity]['per_class'][lcID]['prec'] /= numMatches
                        response[entity]['per_class'][lcID]['rec'] /= numMatches
                        response[entity]['per_class'][lcID]['f1'] /= numMatches

        return {
            'label_classes': labelClasses,
            'per_entity': response
            }


    def getUserAnnotationSpeeds(self, project, users, goldenQuestionsOnly=False):
        '''
            Returns, for each username in "users" list,
            the mean, median and lower and upper quartile
            (25% and 75%) of the time required in a given project.
        '''
        # prepare output
        response = {}
        for u in users:
            response[u] = {
                'avg': float('nan'),
                'median': float('nan'),
                'perc_25': float('nan'),
                'perc_75': float('nan')
            }

        if goldenQuestionsOnly:
            gqStr = sql.SQL('''
                JOIN {id_img} AS img
                ON anno.image = img.id
                WHERE img.isGoldenQuestion = true
            ''').format(
                id_img=sql.Identifier(project, 'image')
            )
        else:
            gqStr = sql.SQL('')

        queryStr = sql.SQL('''
            SELECT username, avg(timeRequired) AS avg,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY timeRequired ASC) AS median,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY timeRequired ASC) AS perc_25,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY timeRequired ASC) AS perc_75
            FROM (
                SELECT username, timeRequired
                FROM {id_anno} AS anno
                {gqStr}
            ) AS q
            WHERE username IN %s
            GROUP BY username
        ''').format(
            id_anno=sql.Identifier(project, 'annotation'),
            gqStr=gqStr
        )
        result = self.dbConnector.execute(queryStr, (tuple(users),), 'all')
        if result is not None:
            for r in result:
                user = r['username']
                response[user] = {
                    'avg': float(r['avg']),
                    'median': float(r['median']),
                    'perc_25': float(r['perc_25']),
                    'perc_75': float(r['perc_75']),
                }
        return response


    def getUserFinished(self, project, username):
        '''
            Returns True if the user has viewed all images in the project,
            and False otherwise.
            We deliberately do not reveal more information to the general
            user, in order to e.g. sustain the golden question limitation
            system.
        '''
        queryStr = sql.SQL('''
            SELECT COUNT(*) AS cnt FROM {id_iu}
            WHERE viewcount > 0 AND username = %s
            UNION ALL
            SELECT COUNT(*) AS cnt FROM {id_img};
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user')
        )
        result = self.dbConnector.execute(queryStr, (username,), 2)
        return result[0]['cnt'] >= result[1]['cnt']


    def getTimeActivity(self, project, type='images', numDaysMax=31, perUser=False):
        '''
            Returns a histogram of the number of images viewed (if type = 'images')
            or annotations made (if type = 'annotations') over the last numDaysMax.
            If perUser is True, statistics are returned on a user basis.
        '''
        if type == 'images':
            id_table = sql.Identifier(project, 'image_user')
            time_field = sql.SQL('last_checked')
        else:
            id_table = sql.Identifier(project, 'annotation')
            time_field = sql.SQL('timeCreated')
        
        if perUser:
            userSpec = sql.SQL(', username')
        else:
            userSpec = sql.SQL('')
        queryStr = sql.SQL('''
            SELECT to_char({time_field}, 'YYYY-Mon-dd') AS month_day, MIN({time_field}) AS date_of_day, COUNT(*) AS cnt {user_spec}
            FROM {id_table}
            WHERE {time_field} IS NOT NULL
            GROUP BY month_day {user_spec}
            ORDER BY date_of_day ASC
            LIMIT %s
        ''').format(
            time_field=time_field,
            id_table=id_table,
            user_spec=userSpec
        )
        result = self.dbConnector.execute(queryStr, (numDaysMax,), 'all')

        #TODO: homogenize series and add missing days

        if perUser:
            response = {}
        else:
            response = {
                'counts': [],
                'timestamps': [],
                'labels': []
            }
            
        for row in result:
            if perUser:
                if row['username'] not in response:
                    response[row['username']] = {
                        'counts': [],
                        'timestamps': [],
                        'labels': []
                    }
                response[row['username']]['counts'].append(row['cnt'])
                response[row['username']]['timestamps'].append(row['date_of_day'].timestamp())
                response[row['username']]['labels'].append(row['month_day'])
            else:
                response['counts'].append(row['cnt'])
                response['timestamps'].append(row['date_of_day'].timestamp())
                response['labels'].append(row['month_day'])
        return response


    def getAccuracy(self, project, entries):
        '''
            For quizgame functionality: compares dict of entries and contained
            annotations with existing annotations in project and returns
            accuracy scores depending on the annotation type.
        '''
        # get project's annotation type
        if project not in self.annotationTypes:
            annoType = self.dbConnector.execute(sql.SQL('''
                SELECT annotationType FROM "aide_admin".project
                WHERE shortname = %s;
            '''), (project,), 1)
            self.annotationTypes[project] = annoType[0]['annotationtype']
        annoType = self.annotationTypes[project]

        # get existing annotations for entries
        imageIDs = [uuid.UUID(key) for key in entries.keys()]
        colnames = getattr(QueryStrings_annotation, annoType)
        target_anno = self.dbConnector.execute(sql.SQL('''
            SELECT {colnames}, i.filename
            FROM {id_anno} AS a
            JOIN {id_img} AS i
            ON a.image = i.id
            WHERE image IN %s
        ''').format(
            colnames=sql.SQL(','.join([f'a.{c}' for c in colnames.value])),
            id_anno=sql.Identifier(project, 'annotation'),
            id_img=sql.Identifier(project, 'image')
        ), tuple((i,) for i in imageIDs), 'all')
        targets = defaultdict(dict)     # used for accuracy evaluation
        targets_out = {}                # sent back to the user as a solution
        for row in target_anno:
            imgID = str(row['image'])
            if 'annotations' not in targets[imgID]:
                targets[imgID]['annotations'] = []
            targets[imgID]['annotations'].append(row)
            
            if imgID not in targets_out:
                targets_out[imgID] = {
                    'fileName': row['filename'],
                    'predictions': {},
                    'annotations': {},
                    'last_checked': None
                }
            entry = {}
            for c in colnames.value:
                if c not in row:
                    entry[c] = None
                    continue
                value = row[c]
                if isinstance(value, datetime):
                    value = value.timestamp()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                entry[c] = value
            annoID = str(row['id']) + str(datetime.now())       # modify ID to avoid conflicts in UI
            targets_out[imgID]['annotations'][annoID] = entry

        # calculate statistics
        #TODO
        stats = accuracy.statistics_boundingBoxes(entries, targets)

        # append targets for comparison in viewer
        stats['targets'] = targets_out

        return stats