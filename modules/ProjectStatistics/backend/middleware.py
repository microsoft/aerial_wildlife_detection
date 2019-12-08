'''
    Middleware layer for project statistics calculations.

    2019 Benjamin Kellenberger
'''

from psycopg2 import sql
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
            SELECT NULL AS username, NULL AS num_img, COUNT(DISTINCT(image)) AS num_anno FROM {id_anno}
            UNION ALL
            SELECT NULL AS username, NULL AS num_img, COUNT(*) AS num_anno FROM {id_anno}
            UNION ALL
            SELECT username, iu_cnt AS num_img, anno_cnt AS num_anno FROM (
            SELECT u.username, iu_cnt, anno_cnt
            FROM (
                SELECT DISTINCT(username) FROM (
                    SELECT username FROM {id_auth}
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
            id_auth=sql.Identifier(project, 'authentication')
        )
        result = self.dbConnector.execute(queryStr, None, 'all')

        response = {
            'num_images': result[0]['num_img'],
            'num_viewed': result[1]['num_img'],
            'num_annotated': result[2]['num_anno'],
            'num_annotations': result[3]['num_anno']
        }
        if len(result) > 4:
            response['user_stats'] = {}
            for i in range(4, len(result)):
                uStats = {
                    'num_viewed': result[i]['num_img'],
                    'num_annotations': result[i]['num_anno']
                }
                response['user_stats'][result[i]['username']] = uStats
        return response