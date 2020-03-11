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

        #     # cap by limit specified in settings
        #     limit = self.config.getProperty('AIController', 'maxNumImages_train', type=int)
        # if minNumAnnoPerImage is None or minNumAnnoPerImage == -1:
        #     minNumAnnoPerImage = self.config.getProperty('AIController', 'minNumAnnoPerImage', type=int)


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

    

    def getInferenceQueryString(self, project, forceUnlabeled=True, limit=None):

        if forceUnlabeled:
            unlabeledString = sql.SQL('WHERE {id_iu}.viewcount IS NULL').format(
                id_iu=sql.Identifier(project, 'image_user')
            )
        else:
            unlabeledString = sql.SQL('')
        
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
                {unlabeledString}
                ORDER BY image_user.viewcount ASC NULLS FIRST
                {limit}
            ) AS query;
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user'),
            unlabeledString=unlabeledString,
            limit=limitString
        )
        return queryStr