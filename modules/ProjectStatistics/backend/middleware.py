'''
    Middleware layer for project statistics calculations.

    2019-20 Benjamin Kellenberger
'''

from psycopg2 import sql
from .statisticalFormulas import StatisticalFormulas_user, StatisticalFormulas_model
from modules.Database.app import Database


class ProjectStatisticsMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
    

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


    def getPerformanceStatistics(self, project, entities_eval, username_groundTruth, entityType='user', threshold=0.5, goldenQuestionsOnly=True):
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
                    TODO: not supported yet

            Value 'threshold' determines the geometric requirement for an annotation to be
            counted as correct (or incorrect) as follows:
                - points: maximum euclidean distance in pixels to closest target
                - bounding boxes: minimum IoU with best matching target

            If 'goldenQuestionsOnly' is True, only images with flag 'isGoldenQuestion' = True
            will be considered for evaluation.
        '''
        entityType = entityType.lower()

        # get annotation type for project
        annoType = self.dbConnector.execute('''SELECT annotationType
            FROM aide_admin.project WHERE shortname = %s;''',
            (project,),
            1)
        annoType = annoType[0]['annotationtype']

        # compose args list and complete query
        queryArgs = [username_groundTruth, tuple(entities_eval)]
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
        tokens = {
            'num_pred': 0,
            'num_target': 0,
        }
        if annoType == 'points':
            tokens = { **tokens, **{
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'avg_dist': 0.0
            }}
        elif annoType == 'boundingBoxes':
            tokens = { **tokens, ** {
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'avg_iou': 0.0
            }}
        
        tokens_normalize = [
            'avg_dist',
            'avg_iou'
        ]
        
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
                id_iu=sql.Identifier(project, 'image_user'),        #TODO: needed for model tests?
                id_cnnstate=sql.Identifier(project, 'cnnstate'),    #TODO: needed? always?
                sql_goldenQuestion=sql_goldenQuestion
            )

        #TODO: update points query (according to bboxes); re-write stats parsing below

        # get stats
        response = {
            'per_entity': {}
        }
        with self.dbConnector.execute_cursor(queryStr, tuple(queryArgs)) as cursor:
            while True:
                b = cursor.fetchone()
                if b is None:
                    break

                if entityType == 'user':
                    entity = b['username']
                else:
                    entity = b['cnnstate']
                if not entity in response['per_entity']:
                    response['per_entity'][entity] = tokens.copy()
                    response['per_entity'][entity]['count'] = 1
                elif b['num_target'] > 0:
                    response['per_entity'][entity]['count'] += 1
                
                for key in tokens.keys():
                    if b[key] is not None:
                        response['per_entity'][entity][key] += b[key]

        for entity in response['per_entity'].keys():
            for t in tokens_normalize:
                if t in response['per_entity'][entity]:
                    response['per_entity'][entity][t] /= response['per_entity'][entity]['count']
            del response['per_entity'][entity]['count']

            if annoType == 'points' or annoType == 'boundingBoxes':
                prec, rec, f1 = self._calc_geometric_stats(
                    response['per_entity'][entity]['tp'],
                    response['per_entity'][entity]['fp'],
                    response['per_entity'][entity]['fn']
                )
                response['per_entity'][entity]['prec'] = prec
                response['per_entity'][entity]['rec'] = rec
                response['per_entity'][entity]['f1'] = f1

        return response