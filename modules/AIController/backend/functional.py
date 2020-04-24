'''
    Definition of AIController tasks that are distributed
    via Celery (e.g. assembling and splitting the lists of
    training and testing images).

    2020 Benjamin Kellenberger
'''

from datetime import datetime
from psycopg2 import sql
from modules.Database.app import Database
from util.helpers import array_split
from .sql_string_builder import SQLStringBuilder



class AIControllerWorker:

    def __init__(self, config, celery_app):
        self.config = config
        self.dbConn = Database(config)
        self.sqlBuilder = SQLStringBuilder(config)
        self.celery_app = celery_app


    
    def _get_num_available_workers(self):
        #TODO: filter for right tasks and queues
        #TODO: limit to n tasks per worker
        i = self.celery_app.control.inspect()
        if i is not None:
            stats = i.stats()
            if stats is not None:
                return len(i.stats())
        return 1    #TODO



    def get_training_images(self, project, epoch=None, minTimestamp='lastState', includeGoldenQuestions=True,
                            minNumAnnoPerImage=0, maxNumImages=None, numChunks=1):
        '''
            Queries the database for the latest images to be used for model training.
            Returns a list with image UUIDs accordingly, split into the number of
            available workers.
            #TODO: includeGoldenQuestions
        '''
        # sanity checks
        if not (isinstance(minTimestamp, datetime) or minTimestamp == 'lastState' or
                minTimestamp == -1 or minTimestamp is None):
            raise ValueError('{} is not a recognized property for variable "minTimestamp"'.format(str(minTimestamp)))

        # query image IDs
        queryVals = []

        if minTimestamp is None:
            timestampStr = sql.SQL('')
        elif minTimestamp == 'lastState':
            timestampStr = sql.SQL('''
            WHERE iu.last_checked > COALESCE(to_timestamp(0),
            (SELECT MAX(timecreated) FROM {id_cnnstate}))''').format(
                id_cnnstate=sql.Identifier(project, 'cnnstate')
            )
        elif isinstance(minTimestamp, datetime):
            timestampStr = sql.SQL('WHERE iu.last_checked > COALESCE(to_timestamp(0), %s)')
            queryVals.append(minTimestamp)
        elif isinstance(minTimestamp, int) or isinstance(minTimestamp, float):
            timestampStr = sql.SQL('WHERE iu.last_checked > COALESCE(to_timestamp(0), to_timestamp(%s))')
            queryVals.append(minTimestamp)

        if minNumAnnoPerImage > 0:
            queryVals.append(minNumAnnoPerImage)

        if maxNumImages is None or maxNumImages < 0:
            limitStr = sql.SQL('')
        else:
            limitStr = sql.SQL('LIMIT %s')
            queryVals.append(maxNumImages)

        if minNumAnnoPerImage <= 0:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE corrupt IS NULL OR corrupt = FALSE
                    ) AS imgQ
                    ON iu.image = imgQ.iid
                    {timestampStr}
                    ORDER BY iu.last_checked ASC
                    {limitStr}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_img=sql.Identifier(project, 'image'),
                timestampStr=timestampStr,
                limitStr=limitStr)

        else:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE corrupt IS NULL OR corrupt = FALSE
                    ) AS imgQ
                    ON iu.image = imgQ.iid
                    {timestampStr}
                    {conjunction} image IN (
                        SELECT image FROM (
                            SELECT image, COUNT(*) AS cnt
                            FROM {id_anno}
                            GROUP BY image
                            ) AS annoCount
                        WHERE annoCount.cnt >= %s
                    )
                    ORDER BY iu.last_checked ASC
                    {limitStr}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_img=sql.Identifier(project, 'image'),
                id_anno=sql.Identifier(project, 'annotation'),
                timestampStr=timestampStr,
                conjunction=(sql.SQL('WHERE') if minTimestamp is None else sql.SQL('AND')),
                limitStr=limitStr)

        imageIDs = self.dbConn.execute(queryStr, tuple(queryVals), 'all')
        imageIDs = [i['image'] for i in imageIDs]

        if numChunks > 1:
            # split for distribution across workers (TODO: also specify subset size for multiple jobs; randomly draw if needed)
            imageIDs = array_split(imageIDs, max(1, len(imageIDs) // numChunks))
        else:
            imageIDs = [imageIDs]

        print("Assembled training images into {} chunks (length of first: {})".format(len(imageIDs), len(imageIDs[0])))
        return imageIDs



    def get_inference_images(self, project, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, numChunks=1):
            '''
                Queries the database for the latest images to be used for inference after model training.
                Returns a list with image UUIDs accordingly, split into the number of available workers.
                #TODO: goldenQuestionsOnly
            '''
            if maxNumImages is None or maxNumImages < 0:
                queryResult = self.dbConn.execute('''
                    SELECT maxNumImages_inference
                    FROM aide_admin.project
                    WHERE shortname = %s;''', (project,), 1)
                maxNumImages = queryResult['maxnumimages_inference']    
            
            queryVals = (maxNumImages,)

            # load the IDs of the images that are being subjected to inference
            sql = self.sqlBuilder.getInferenceQueryString(project, forceUnlabeled, maxNumImages)
            imageIDs = self.dbConn.execute(sql, queryVals, 'all')
            imageIDs = [i['image'] for i in imageIDs]

            # # split for distribution across workers
            # if maxNumWorkers != 1:
            #     # only query the number of available workers if more than one is specified to save time
            #     num_available = self._get_num_available_workers()
            #     if maxNumWorkers == -1:
            #         maxNumWorkers = num_available   #TODO: more than one process per worker?
            #     else:
            #         maxNumWorkers = min(maxNumWorkers, num_available)
            
            if numChunks > 1:
                imageIDs = array_split(imageIDs, max(1, len(imageIDs) // numChunks))
            else:
                imageIDs = [imageIDs]
            return imageIDs