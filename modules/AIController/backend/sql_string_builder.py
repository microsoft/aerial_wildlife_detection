'''
    SQL string builder for AIController.

    2019 Benjamin Kellenberger
'''

from datetime import datetime

class SQLStringBuilder:

    def __init__(self, config):
        self.config = config

    
    def getFixedImageIDQueryString(self, ids):
        pass

    
    def getTimestampQueryString(self, timestamp, order='oldest', limit=None):
        
        # parse timestamp given
        if timestamp == 'lastState':
            # select every image that has been labeled after the last model state
            timestampSpecifier = 'MAX(timecreated)'
        
        elif isinstance(timestamp, datetime):
            timestampSpecifier = '%s'

        else:
            timestampSpecifier = 'to_timestamp(0)'  #TODO: lazy hack...


        if limit is None or limit == -1:
            # cap by limit specified in settings
            limit = self.config.getProperty('AIController', 'maxNumImages_train')
            
        else:
            limit = min(limit, self.config.getProperty('AIController', 'maxNumImages_train'))

        if order == 'oldest':
            order = 'ASC'
        else:
            order = 'DESC'

        sql = '''
            SELECT newestAnno.image FROM (
                SELECT image, timecreated FROM {schema}.annotation AS anno
                WHERE anno.timecreated > (SELECT COALESCE({tsSpec}, to_timestamp(0)) AS latestState FROM {schema}.cnnstate)
                ORDER BY anno.timecreated {order}
                LIMIT {limit}
            ) AS newestAnno;
        '''.format(schema=self.config.getProperty('Database', 'schema'),
                    tsSpec=timestampSpecifier, order=order, limit=limit)
        return sql

    

    def getInferenceQueryString(self, forceUnlabeled=True, limit=None):

        if forceUnlabeled:
            unlabeledString = 'WHERE image_user.viewcount IS NULL'
        else:
            unlabeledString = ''
        
        if limit is None or limit == -1:
            limitString = ''
        elif isinstance(limit, int):
            limitString = 'LIMIT {}'.format(limit)
        else:
            raise ValueError('Invalid value for limit ({})'.format(limit))

        sql = '''
            SELECT query.imageID, query.viewcount FROM (
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