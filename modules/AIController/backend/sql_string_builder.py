'''
    SQL string builder for AIController.

    2019 Benjamin Kellenberger
'''

from datetime import datetime

class SQLStringBuilder:

    def __init__(self, config):
        self.config = config

    
    def getFixedImageIDQueryString(self, project, ids):
        pass

    
    def getLatestQueryString(self, project, minNumAnnoPerImage=0, limit=None):
        #TODO: re-formulate for project
        if limit is None or limit == -1:
            # cap by limit specified in settings
            limit = self.config.getProperty('AIController', 'maxNumImages_train', type=int)
        if minNumAnnoPerImage is None or minNumAnnoPerImage == -1:
            minNumAnnoPerImage = self.config.getProperty('AIController', 'minNumAnnoPerImage', type=int)

        if minNumAnnoPerImage <= 0:
            # no restriction on number of annotations per image
            sql = '''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {schema}.image_user AS iu
                    -- WHERE iu.last_checked > COALESCE(to_timestamp(0), (SELECT MAX(timecreated) FROM {schema}.cnnstate))
                    ORDER BY iu.last_checked ASC
                    LIMIT {limit}
                ) AS newestAnno;
            '''.format(schema=self.config.getProperty('Database', 'schema'),
                    limit=limit)
        else:
            sql = '''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {schema}.image_user AS iu
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
            '''.format(schema=self.config.getProperty('Database', 'schema'),
                    minAnnoCount=minNumAnnoPerImage,
                    limit=limit)
        return sql

    

    def getInferenceQueryString(self, forceUnlabeled=True, limit=None):

        if forceUnlabeled:
            unlabeledString = 'WHERE image_user.viewcount IS NULL'
        else:
            unlabeledString = ''
        
        if limit is None or limit == -1:
            limitString = ''
        else:
            try:
                limitString = 'LIMIT {}'.format(int(limit))
            except:
                raise ValueError('Invalid value for limit ({})'.format(limit))

        sql = '''
            SELECT query.imageID AS image FROM (
                SELECT image.id AS imageID, image_user.viewcount FROM {schema}.image
                LEFT OUTER JOIN {schema}.image_user
                ON image.id = image_user.image
                {unlabeledString}
                ORDER BY image_user.viewcount ASC NULLS FIRST
                {limit}
            ) AS query;
        '''.format(schema=self.config.getProperty('Database', 'schema'),
            unlabeledString=unlabeledString,
            limit=limitString
        )
        return sql