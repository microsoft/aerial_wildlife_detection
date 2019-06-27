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

        baseNames += ['id']

        return baseNames


    def getFixedImagesQueryString(self):
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('LabelUI', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('AITrainer', 'annotationType')).value
        fields_union = list(fields_anno.union(fields_pred))
        string_anno = ''
        string_pred = ''
        for f in fields_union:
            if not f in fields_anno:
                string_anno += 'NULL AS '
            if not f in fields_pred:
                string_pred += 'NULL AS '
            string_anno += f + ','
            string_pred += f + ','
        string_anno = string_anno.strip(',')
        string_pred = string_pred.strip(',')

        #TODO: username
        sql = '''
            SELECT * FROM (
                SELECT id, image, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                UNION ALL
                SELECT id, image, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents
            JOIN (
                SELECT id AS imageID, filename FROM {schema}.image
            ) AS img ON img.imageID = contents.image
            WHERE contents.image IN ( %s );
        '''.format(schema=schema, annoCols=string_anno, predCols=string_pred)
        return sql


    def getNextBatchQueryString(self, subset='preferUnlabeled'):
        '''
            Assembles a DB query string. Inputs:
            - subset: specifies how to constrain the returned set of
              images, annotations and predictions. One of:
                - preferUnlabeled: prioritizes images that have not (yet)
                  been seen by the user (i.e., viewcount is None or zero)
                - forceUnlabeled: only show images that have no/zero viewcount.
                  This may result in an empty set.
                - preferLabeled: prioritize images with a non-None/non-zero view-
                  count. However, viewcounts are sorted in ascending order.
                - preferLabeledDesc: prioritize images according to viewcount in
                  descending order
                - forceLabeled: restrict to images whose viewcount is one or more,
                  sorted by viewcount in ascending order.
                  May result in an empty set.
                - forceLabeledDesc: restrict to seen images, sorted by viewcount in
                  descending order.
        '''
        schema = self.config.getProperty('Database', 'schema')

        # assemble column names
        fields_anno = getattr(FieldNames_annotation, self.config.getProperty('LabelUI', 'annotationType')).value
        fields_pred = getattr(FieldNames_prediction, self.config.getProperty('AITrainer', 'annotationType')).value
        fields_union = list(fields_anno.union(fields_pred))
        string_anno = ''
        string_pred = ''
        for f in fields_union:
            if not f in fields_anno:
                string_anno += 'NULL AS '
            if not f in fields_pred:
                string_pred += 'NULL AS '
            string_anno += f + ','
            string_pred += f + ','
        string_anno = string_anno.strip(',')
        string_pred = string_pred.strip(',')

        # subset selection fragment
        subsetFragment = ''
        if subset == 'preferUnlabeled':
            subsetFragment = '''
                ORDER BY img_user.viewcount ASC NULLS FIRST, score DESC
            '''
        elif subset == 'forceUnlabeled':
            subsetFragment = '''
                HAVING img_user.viewcount IS NULL or img_user.viewcount = 0
                ORDER BY img_user.viewcount ASC NULLS FIRST, score DESC
            '''
        elif subset == 'preferLabeled':
            subsetFragment = '''
                ORDER BY img_user.viewcount ASC NULLS LAST, score DESC
            '''
        elif subset == 'preferLabeledDesc':
            subsetFragment = '''
                ORDER BY img_user.viewcount DESC NULLS LAST, score DESC
            '''
        elif subset == 'forceLabeled':
            subsetFragment = '''
                HAVING img_user.viewcount > 0
                ORDER BY img_user.viewcount ASC NULLS LAST, score DESC
            '''
        elif subset == 'forceLabeledDesc':
            subsetFragment = '''
                HAVING img_user.viewcount > 0
                ORDER BY img_user.viewcount DESC NULLS LAST, score DESC
            '''
        else:
            raise ValueError('{} is not a recognized subset specifier.'.format(subset))

        sql = '''
            SELECT * FROM (
                SELECT id, image, 'annotation' AS cType, {annoCols} FROM {schema}.annotation AS anno
                UNION ALL
                SELECT id, image, 'prediction' AS cType, {predCols} FROM {schema}.prediction AS pred
            ) AS contents
            JOIN (
                SELECT id AS imageID, filename FROM {schema}.image
            ) AS img ON img.imageID = contents.image
            WHERE contents.image IN (
                SELECT topK.image FROM (
                    SELECT pred.image, sum(confidence)/count(confidence) AS score, img_user.viewcount FROM aerialelephants.prediction AS pred
                    FULL OUTER JOIN (
                        SELECT * FROM aerialelephants.image_user WHERE username = %s
                    ) AS img_user ON pred.image = img_user.image
                    GROUP BY pred.image, img_user.viewcount
                    {subset}
                    LIMIT %s
                ) AS topK
            );
        '''.format(schema=schema, annoCols=string_anno, predCols=string_pred, subset=subsetFragment)

        return sql