'''
    Factory that creates SQL strings for querying and submission,
    adjusted to the arguments specified.

    2019 Benjamin Kellenberger
'''
from enum import Enum

class FieldNames_prediction(Enum):
    labels = set(['label', 'confidence'])
    points = set(['label', 'confidence', 'x', 'y'])
    boundingBoxes = set(['label', 'confidence', 'x', 'y', 'width', 'height'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename

class FieldNames_annotation(Enum):
    labels = set(['label'])
    points = set(['label', 'x', 'y'])
    boundingBoxes = set(['label', 'x', 'y', 'width', 'height'])
    segmentationMasks = set(['filename'])   #TODO: conflict with image filename


class SQLStringBuilder:

    def __init__(self, config):
        self.config = config


    def getColnames(self, type):
        '''
            Returns a list of column names, depending on the type specified
            (either 'prediction' or 'annotation').
        '''
        if type == 'prediction':
            baseNames = list(getattr(FieldNames_prediction, self.config.getProperty('AITrainer', 'annotationType')).value)
        elif type == 'annotation':
            baseNames = list(getattr(FieldNames_annotation, self.config.getProperty('LabelUI', 'annotationType')).value)
        else:
            raise ValueError('{} is not a recognized type.'.format(type))

        baseNames += ['id', 'viewcount']

        return baseNames


    def getFixedImagesQueryString(self):
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('LabelUI', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('AITrainer', 'annotationType')).value
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

        sql = '''
            SELECT id, image, cType, viewcount, filename, {allCols} FROM (
                SELECT id AS image, filename FROM {schema}.image
                WHERE id IN %s
            ) AS img
            LEFT OUTER JOIN (
                SELECT id, image AS imID, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                WHERE username = %s
                UNION ALL
                SELECT id, image AS imID, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents ON img.image = contents.imID
            LEFT OUTER JOIN (SELECT image AS iu_image, viewcount, username FROM {schema}.image_user
            WHERE username = %s) AS iu ON img.image = iu.iu_image;
        '''.format(schema=schema, allCols=string_all, annoCols=string_anno, predCols=string_pred)
        return sql


    def getNextBatchQueryString(self, order='unlabeled', subset='default'):
        '''
            Assembles a DB query string. Inputs:
            - order: specifies sorting criterion for request:
                - 'unlabeled': prioritize images that have not (yet) been viewed
                    by the current user (i.e., zero/low viewcount)
                - 'labeled': put images first in order that have a high user viewcount
            - subset: hard constraint on the label status of the images:
                - 'default': do not constrain query set
                - 'forceLabeled': images must have a viewcount of 1 or more
                - 'forceUnlabeled': images must not have been viewed by the current user
        '''
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('LabelUI', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('AITrainer', 'annotationType')).value
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
        orderSpec = ''
        if subset == 'forceLabeled':
            subsetFragment = 'WHERE viewcount > 0'
        elif subset == 'forceUnlabeled':
            subsetFragment = 'WHERE viewcount IS NULL OR viewcount = 0'

        if order == 'unlabeled':
            orderSpec = 'ORDER BY viewcount ASC NULLS FIRST, score DESC'
        elif order == 'labeled':
            orderSpec = 'ORDER BY viewcount DESC NULLS LAST, score DESC'


        sql = '''
            SELECT id, image, cType, viewcount, filename, {allCols} FROM (
            SELECT id AS image, filename, viewcount, score FROM {schema}.image AS img
            LEFT OUTER JOIN (
                SELECT * FROM {schema}.image_user
                WHERE username = %s
            ) AS iu ON img.id = iu.image
            LEFT OUTER JOIN (
                SELECT image, SUM(confidence)/COUNT(confidence) AS score
                FROM {schema}.prediction
                GROUP BY image
            ) AS img_score ON img.id = img_score.image
            {subset}
            {order}
            LIMIT %s
            ) AS img_query
            LEFT OUTER JOIN (
                SELECT id, image AS imID, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                WHERE username = %s
                UNION ALL
                SELECT id, image AS imID, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents ON img_query.image = contents.imID
            {order};
        '''.format(schema=schema, allCols=string_all, annoCols=string_anno, predCols=string_pred, order=orderSpec, subset=subsetFragment)

        return sql