'''
    SQL string builder for AIController.

    2019-20 Benjamin Kellenberger
'''

from psycopg2 import sql


class SQLStringBuilder:

    def __init__(self, config):
        self.config = config

    
    # def getFixedImageIDQueryString(self, project, ids):
    #     #TODO: implement
    #     pass

    
    def getLatestQueryString(self, project, minNumAnnoPerImage=0, limit=None):
        if limit is None or limit == -1:
            limitStr = sql.SQL('')
        else:
            limitStr = sql.SQL('LIMIT %s')

        if minNumAnnoPerImage <= 0:
            # no restriction on number of annotations per image
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    -- WHERE iu.last_checked > COALESCE(to_timestamp(0), (SELECT MAX(timecreated) FROM {id_cnnstate}))
                    ORDER BY iu.last_checked ASC
                    {limit}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_cnnstate=sql.Identifier(project, 'cnnstate'),
                limit=limitStr)
        else:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    WHERE image IN (
                        SELECT image FROM (
                            SELECT image, COUNT(*) AS cnt
                            FROM {schema}.annotation
                            GROUP BY image
                            ) AS annoCount
                        WHERE annoCount.cnt > {minAnnoCount}
                    )
                    ORDER BY iu.last_checked ASC
                    LIMIT {limit}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                minAnnoCount=minNumAnnoPerImage,
                limit=limit)
        return queryStr

    

    def getInferenceQueryString(self, project, forceUnlabeled=True, goldenQuestionsOnly=False, limit=None):

        if goldenQuestionsOnly:
            gqString = sql.SQL('AND goldenQuestion IS NOT NULL')
        else:
            gqString = sql.SQL('')

        if forceUnlabeled:
            conditionString = sql.SQL('WHERE viewcount IS NULL AND (corrupt IS NULL OR corrupt = FALSE) {}').format(gqString)
        else:
            conditionString = sql.SQL('WHERE corrupt IS NULL OR corrupt = FALSE {}').format(gqString)
        
        if limit is None or limit == -1:
            limitString = sql.SQL('')
        else:
            try:
                limitString = sql.SQL('LIMIT %s')
            except:
                raise ValueError('Invalid value for limit ({})'.format(limit))

        queryStr = sql.SQL('''
            SELECT query.imageID AS image FROM (
                SELECT image.id AS imageID, image_user.viewcount FROM {id_img}
                LEFT OUTER JOIN {id_iu}
                ON image.id = image_user.image
                {conditionString}
                ORDER BY image_user.viewcount ASC NULLS FIRST
                {limit}
            ) AS query;
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user'),
            conditionString=conditionString,
            limit=limitString
        )
        return queryStr