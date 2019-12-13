'''
    Middleware layer for project statistics calculations.

    2019 Benjamin Kellenberger
'''

from psycopg2 import sql
from .statisticalFormulas import StatisticalFormulas
from modules.Database.app import Database


class ProjectStatisticsMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
    

    def getProjectStatistics(self, project):
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


    @staticmethod
    def _calc_geometric_stats(tp, fp, fn):
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


    def getUserStatistics(self, project, username_eval, username_groundTruth, threshold=0.5, goldenQuestionsOnly=True, perImage=False):
        '''
            Compares the accuracy of a user with a second one.
            The following measures of accuracy are reported, depending on the
            annotation type:            TODO: enable prediction accuracy too
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
            If 'perImage' is True, statistical figures will be provided for each image.
            Else only general values, averaged over the entire set of images, are returned.
        '''

        # get annotation type for project
        annoType = self.dbConnector.execute('''SELECT annotationType
            FROM aide_admin.project WHERE shortname = %s;''',
            (project,),
            1)
        annoType = annoType[0]['annotationtype']

        # compose args list and complete query
        queryArgs = [username_eval, username_groundTruth]

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
        
        
        if annoType == 'points' or annoType == 'boundingBoxes':
            queryArgs.extend([threshold, threshold])
            queryArgs.extend([username_eval, username_groundTruth])

            if not perImage:
                sql_global_start = sql.SQL('''SELECT SUM(num_pred) AS num_pred, SUM(num_target) AS num_target,
                    SUM(ntp) AS ntp, SUM(nfp) AS nfp, SUM(nfn) AS nfn
                    FROM (''')
                sql_global_end = sql.SQL(') AS globalQuery;')

            else:
                sql_global_start = sql.SQL('')
                sql_global_end = sql.SQL(';')

        elif annoType == 'segmentationMasks':
            raise Exception('Per-user statistics on segmentation masks are not yet supported.')
    
        elif annoType == 'labels':
            if not perImage:
                sql_global_start = sql.SQL('SELECT SUM(label_correct::int) AS num_correct, COUNT(*) AS num_total FROM (')
                sql_global_end = sql.SQL(') AS globalQuery;')

            else:
                sql_global_start = sql.SQL('')
                sql_global_end = sql.SQL(';')

        queryStr = getattr(StatisticalFormulas, annoType).value
        queryStr = sql.SQL(queryStr).format(
            id_anno=sql.Identifier(project, 'annotation'),
            id_iu=sql.Identifier(project, 'image_user'),
            sql_goldenQuestion=sql_goldenQuestion,
            sql_global_start=sql_global_start,
            sql_global_end=sql_global_end
        )

        # get stats
        response = {}

        if perImage:
            response['per_image'] = {}
            with self.dbConnector.execute_cursor(queryStr, tuple(queryArgs)) as cursor:
                if annoType == 'points' or annoType == 'boundingBoxes':
                    global_stats = {
                        'num_pred': 0,
                        'num_target': 0,
                        'tp': 0,
                        'fp': 0,
                        'fn': 0
                    }
                else:
                    global_stats = {
                        'num_correct': 0,
                        'num_total': 0,
                        'oa': 0.0
                    }
                
                # iterate over results
                while True:
                    b = cursor.fetchone()
                    if b is None:
                        break

                    if annoType == 'points' or annoType == 'boundingBoxes':
                        prec, rec, f1 = self._calc_geometric_stats(b['ntp'], b['nfp'], b['nfn'])
                        response['per_image'][str(b['image'])] = {
                            'num_pred': b['num_pred'],
                            'num_target': b['num_target'],
                            'tp': b['ntp'],
                            'fp': b['nfp'],
                            'fn': b['nfn'],
                            'precision': prec,
                            'recall': rec,
                            'f1': f1
                        }
                        global_stats['num_pred'] += b['num_pred']
                        global_stats['num_target'] += b['num_target']
                        global_stats['tp'] += b['ntp']
                        global_stats['fp'] += b['nfp']
                        global_stats['fn'] += b['nfn']

                    else:
                        try:
                            oa = 100 * b['num_correct'] / b['num_total']
                        except:
                            oa = None
                        response['per_image'][str(b['image'])] = {
                            'num_correct': b['num_correct'],
                            'num_total': b['num_total'],
                            'oa': oa
                        }
                        global_stats['num_correct'] += b['num_correct']
                        global_stats['num_total'] += b['num_total']

            if annoType == 'points' or annoType == 'boundingBoxes':
                prec, rec, f1 = self._calc_geometric_stats(global_stats['tp'], global_stats['fp'], global_stats['fn'])
                global_stats['precision'] = prec
                global_stats['recall'] = rec
                global_stats['f1'] = f1

            else:
                try:
                    oa = 100 * global_stats['num_correct'] / global_stats['num_total']
                except:
                    oa = None
                global_stats['oa'] = oa
                
            response['global_stats'] = global_stats

        else:
            # get global stats directly
            result = self.dbConnector.execute(queryStr, tuple(queryArgs), 1)
            result = result[0]

            if annoType == 'points' or annoType == 'boundingBoxes':
                prec, rec, f1 = self._calc_geometric_stats(result['ntp'], result['nfp'], result['nfn'])
                response = {
                    'num_pred': result['num_pred'],
                    'num_target': result['num_target'],
                    'tp': result['ntp'],
                    'fp': result['nfp'],
                    'fn': result['nfn'],
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                }

            else:
                try:
                    oa = 100 * result['num_correct'] / result['num_total']
                except:
                    oa = None
                response = {
                    'num_correct': result['num_correct'],
                    'num_total': result['num_total'],
                    'oa': oa
                }
            
        return response