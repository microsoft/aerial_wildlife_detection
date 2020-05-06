'''
    Factory that creates SQL strings for querying and submission,
    adjusted to the arguments specified.

    2019-20 Benjamin Kellenberger
'''

from constants.dbFieldNames import FieldNames_annotation, FieldNames_prediction


class SQLStringBuilder:

    def __init__(self, config):
        self.config = config


    def getColnames(self, type):
        '''
            Returns a list of column names, depending on the type specified
            (either 'prediction' or 'annotation').
        '''
        if type == 'prediction':
            baseNames = list(getattr(FieldNames_prediction, self.config.getProperty('Project', 'predictionType')).value)
        elif type == 'annotation':
            baseNames = list(getattr(FieldNames_annotation, self.config.getProperty('Project', 'annotationType')).value)
        else:
            raise ValueError('{} is not a recognized type.'.format(type))

        baseNames += ['id', 'viewcount']
        
        return baseNames


    def getFixedImagesQueryString(self, demoMode=False):
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('Project', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('Project', 'predictionType')).value
        fields_union = list(fields_anno.union(fields_pred))
        string_anno = ''
        string_pred = ''
        string_all = ''
        for f in fields_union:
            if not f in fields_anno:
                string_anno += 'NULL AS '
            if not f in fields_pred:
                string_pred += 'NULL AS '
            string_anno += f + ','
            string_pred += f + ','
            string_all += f + ','
        string_anno = string_anno.strip(',')
        string_pred = string_pred.strip(',')
        string_all = string_all.strip(',')

        usernameString = 'WHERE username = %s'
        if demoMode:
            usernameString = ''

        sql = '''
            SELECT id, image, cType, viewcount, EXTRACT(epoch FROM last_checked) as last_checked, filename, {allCols} FROM (
                SELECT id AS image, filename FROM {schema}.image
                WHERE id IN %s
            ) AS img
            LEFT OUTER JOIN (
                SELECT id, image AS imID, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                {usernameString}
                UNION ALL
                SELECT id, image AS imID, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents ON img.image = contents.imID
            LEFT OUTER JOIN (SELECT image AS iu_image, viewcount, last_checked, username FROM {schema}.image_user
            {usernameString}) AS iu ON img.image = iu.iu_image;
        '''.format(schema=schema, allCols=string_all, annoCols=string_anno, predCols=string_pred, usernameString=usernameString)
        return sql

    
    def getNextBatchQueryString(self, order='unlabeled', subset='default', demoMode=False):
        '''
            Assembles a DB query string according to the AL and viewcount ranking criterion.
            Inputs:
            - order: specifies sorting criterion for request:
                - 'unlabeled': prioritize images that have not (yet) been viewed
                    (i.e., last_requested timestamp is None or older than 15 minutes
                    (TODO: make project-specific parameter?))
                    by the current user (i.e., zero/low viewcount)
                - 'labeled': put images first in order that have a high user viewcount
            - subset: hard constraint on the label status of the images:
                - 'default': do not constrain query set
                - 'forceLabeled': images must have a viewcount of 1 or more
                - 'forceUnlabeled': images must not have been viewed by the current user
            - demoMode: set to True to disable sorting criterion and return images in random
                        order instead.
        '''
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('Project', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('Project', 'predictionType')).value
        fields_union = list(fields_anno.union(fields_pred))
        string_anno = ''
        string_pred = ''
        string_all = ''
        for f in fields_union:
            if not f in fields_anno:
                string_anno += 'NULL AS '
            if not f in fields_pred:
                string_pred += 'NULL AS '
            string_anno += f + ','
            string_pred += f + ','
            string_all += f + ','
        string_anno = string_anno.strip(',')
        string_pred = string_pred.strip(',')
        string_all = string_all.strip(',')

        # subset selection fragment
        subsetFragment = ''
        subsetFragment_b = ''
        orderSpec = ''
        if subset == 'forceLabeled':
            subsetFragment = 'WHERE viewcount > 0'
        elif subset == 'forceUnlabeled':
            subsetFragment = 'WHERE viewcount IS NULL OR viewcount = 0'
        subsetFragment_b = subsetFragment

        if len(subsetFragment):
            subsetFragment += ' AND img.last_requested IS NULL OR (NOW() - img.last_requested) > interval \'900 second\''
        else:
            subsetFragment = 'WHERE img.last_requested IS NULL OR (NOW() - img.last_requested) > interval \'900 second\''

        if order == 'unlabeled':
            orderSpec = 'ORDER BY viewcount ASC NULLS FIRST, annoCount ASC NULLS FIRST, score DESC NULLS LAST'
        elif order == 'labeled':
            orderSpec = 'ORDER BY viewcount DESC NULLS LAST, score DESC NULLS LAST'
        elif order == 'random':
            orderSpec = 'ORDER BY RANDOM()'
        orderSpec += ', timeCreated DESC'

        usernameString = 'WHERE username = %s'
        if demoMode:
            usernameString = ''
            orderSpec = 'ORDER BY RANDOM()'

        sql = '''
            SELECT id, image, cType, viewcount, EXTRACT(epoch FROM last_checked) as last_checked, filename, {allCols} FROM (
            SELECT id AS image, filename, viewcount, annoCount, last_checked, score, timeCreated FROM {schema}.image AS img
            LEFT OUTER JOIN (
                SELECT * FROM {schema}.image_user
            ) AS iu ON img.id = iu.image
            LEFT OUTER JOIN (
                SELECT image, SUM(confidence)/COUNT(confidence) AS score, timeCreated
                FROM {schema}.prediction
                GROUP BY image, timeCreated
            ) AS img_score ON img.id = img_score.image
            LEFT OUTER JOIN (
				SELECT image, COUNT(*) AS annoCount
				FROM {schema}.annotation
				{usernameString}
				GROUP BY image
			) AS anno_score ON img.id = anno_score.image
            {subset}

            {order}
            LIMIT %s
            ) AS img_query
            LEFT OUTER JOIN (
                SELECT id, image AS imID, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                {usernameString}
                UNION ALL
                SELECT id, image AS imID, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents ON img_query.image = contents.imID
            {subset_b}
            {order};
        '''.format(schema=schema,
                    allCols=string_all,
                    annoCols=string_anno,
                    predCols=string_pred,
                    order=orderSpec,
                    subset=subsetFragment,
                    subset_b=subsetFragment_b,
                    usernameString=usernameString)

        return sql


    def getDateQueryString(self, minAge, maxAge, userNames, skipEmptyImages):
        '''
            Assembles a DB query string that returns images between a time range.
            Useful for reviewing existing annotations.
            Inputs:
            - minAge: earliest timestamp on which the image(s) have been viewed.
                      Set to None to leave unrestricted.
            - maxAge: latest timestamp of viewing (None = unrestricted).
            - userNames: user names to filter the images to. If string, only images
                         viewed by this respective user are returned. If list, the
                         images are filtered according to any of the names within.
                         If None, no user restriction is placed.
            - skipEmptyImages: if True, images without an annotation will be ignored.
        '''
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('Project', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('Project', 'predictionType')).value
        fields_union = list(fields_anno.union(fields_pred))
        string_anno = ''
        string_pred = ''
        string_all = ''
        for f in fields_union:
            if not f in fields_anno:
                string_anno += 'NULL AS '
            if not f in fields_pred:
                string_pred += 'NULL AS '
            string_anno += f + ','
            string_pred += f + ','
            string_all += f + ','
        string_anno = string_anno.strip(',')
        string_pred = string_pred.strip(',')
        string_all = string_all.strip(',')

        # user names
        usernameString = ''
        if userNames is not None:
            if isinstance(userNames, str):
                usernameString = 'WHERE username = %s'
            elif isinstance(userNames, list):
                usernameString = 'WHERE username IN %s'
            else:
                raise Exception('Invalid property for user names')

        # date range
        timestampString = None
        if minAge is not None:
            prefix = ('WHERE' if usernameString == '' else 'AND')
            timestampString = '{} last_checked::TIMESTAMP > TO_TIMESTAMP(%s)'.format(prefix)
        if maxAge is not None:
            prefix = ('WHERE' if (usernameString == '' and timestampString == '') else 'AND')
            timestampString += ' {} last_checked::TIMESTAMP <= TO_TIMESTAMP(%s)'.format(prefix)

        

        # empty images
        skipEmptyString = ''
        if skipEmptyImages:
            skipEmptyString = '''
            AND image IN (
                SELECT image FROM {schema}.annotation
                {usernameString}
            )
            '''.format(schema=schema,
                usernameString=usernameString)

        # SQL string
        sql = '''
            SELECT id, image, cType, username, viewcount, EXTRACT(epoch FROM last_checked) as last_checked, filename, {allCols} FROM (
                SELECT id AS image, filename FROM {schema}.image
            ) AS img
            JOIN (SELECT image AS iu_image, viewcount, last_checked, username FROM {schema}.image_user
            {usernameString}
            {timestampString}
            {skipEmptyString}
            ORDER BY last_checked ASC
            LIMIT %s) AS iu ON img.image = iu.iu_image
            LEFT OUTER JOIN (
                SELECT id, image AS imID, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                {usernameString}
            ) AS contents ON img.image = contents.imID;
        '''.format(schema=schema, allCols=string_all, annoCols=string_anno, predCols=string_pred,
                usernameString=usernameString,
                timestampString=timestampString,
                skipEmptyString=skipEmptyString)
        return sql      #TODO: verify whether query is right


    def getTimeRangeQueryString(self, userNames, skipEmptyImages):
        '''
            Assembles a DB query string that returns a minimum and maximum timestamp
            between which the image(s) have been annotated.
            Inputs:
            - userNames: user names to filter the images to. If string, only images
                         viewed by this respective user are considered. If list, the
                         images are filtered according to any of the names within.
                         If None, no user restriction is placed.
            - skipEmptyImages: if True, images without an annotation will be ignored.
        '''
        schema = self.config.getProperty('Database', 'schema')

        # params
        usernameString = ''
        if userNames is not None:
            if isinstance(userNames, str):
                usernameString = 'WHERE username = %s'
            elif isinstance(userNames, list):
                usernameString = 'WHERE username IN %s'
            else:
                raise Exception('Invalid property for user names')
            skipEmptyString = 'AND'


        skipEmptyString = ''
        if skipEmptyImages:
            skipEmptyString = '''
            JOIN {schema}.annotation AS anno ON iu.image = anno.image
            '''.format(schema=schema)

        sql = '''
            SELECT EXTRACT(epoch FROM MIN(last_checked)) AS minTimestamp, EXTRACT(epoch FROM MAX(last_checked)) AS maxTimestamp
            FROM (
                SELECT iu.image, last_checked FROM {schema}.image_user AS iu
                {usernameString}
                {skipEmptyString}
            ) AS query;
        '''.format(schema=schema,
            usernameString=usernameString,
            skipEmptyString=skipEmptyString)

        return sql