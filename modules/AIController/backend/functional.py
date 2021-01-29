'''
    Definition of AIController tasks that are distributed
    via Celery (e.g. assembling and splitting the lists of
    training and testing images).

    2020-21 Benjamin Kellenberger
'''

from collections import defaultdict
from collections.abc import Iterable
import json
from uuid import UUID
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



    def get_training_images(self, project, epoch=None, numEpochs=None, minTimestamp='lastState', includeGoldenQuestions=True,
                            minNumAnnoPerImage=0, maxNumImages=None, numChunks=1):
        '''
            Queries the database for the latest images to be used for model training.
            Returns a list with image UUIDs accordingly, split into the number of
            available workers.
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

        if maxNumImages is None or maxNumImages <= 0:
            limitStr = sql.SQL('')
        else:
            limitStr = sql.SQL('LIMIT %s')
            queryVals.append(maxNumImages)

        # golden questions
        if includeGoldenQuestions:
            gqStr = sql.SQL('')
        else:
            gqStr = sql.SQL('AND isGoldenQuestion IS NOT TRUE')

        if minNumAnnoPerImage <= 0:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE corrupt IS NULL OR corrupt = FALSE {gqStr}
                    ) AS imgQ
                    ON iu.image = imgQ.iid
                    {timestampStr}
                    ORDER BY iu.last_checked ASC
                    {limitStr}
                ) AS newestAnno;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_img=sql.Identifier(project, 'image'),
                gqStr=gqStr,
                timestampStr=timestampStr,
                limitStr=limitStr)

        else:
            queryStr = sql.SQL('''
                SELECT newestAnno.image FROM (
                    SELECT image, last_checked FROM {id_iu} AS iu
                    JOIN (
                        SELECT id AS iid
                        FROM {id_img}
                        WHERE corrupt IS NULL OR corrupt = FALSE {gqStr}
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
                gqStr=gqStr,
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



    def get_inference_images(self, project, epoch=None, numEpochs=None, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, numChunks=1):
            '''
                Queries the database for the latest images to be used for inference after model training.
                Returns a list with image UUIDs accordingly, split into the number of available workers.
            '''
            if maxNumImages is None or maxNumImages <= 0:
                queryResult = self.dbConn.execute('''
                    SELECT maxNumImages_inference
                    FROM aide_admin.project
                    WHERE shortname = %s;''', (project,), 1)
                maxNumImages = queryResult[0]['maxnumimages_inference']    
            
            queryVals = (maxNumImages,)

            # load the IDs of the images that are being subjected to inference
            sql = self.sqlBuilder.getInferenceQueryString(project, forceUnlabeled, goldenQuestionsOnly, maxNumImages)
            imageIDs = self.dbConn.execute(sql, queryVals, 'all')
            imageIDs = [i['image'] for i in imageIDs]
            
            if numChunks > 1:
                imageIDs = array_split(imageIDs, max(1, len(imageIDs) // numChunks))
            else:
                imageIDs = [imageIDs]
            return imageIDs


    
    def delete_model_states(self, project, modelStateIDs):
        '''
            Deletes model states with provided IDs from the database
            for a given project.
        '''
        # verify IDs
        if not isinstance(modelStateIDs, Iterable):
            modelStateIDs = [modelStateIDs]
        uuids = []
        modelIDs_invalid = []
        for m in modelStateIDs:
            try:
                uuids.append((UUID(m),))
            except:
                modelIDs_invalid.append(str(m))
        self.dbConn.execute(sql.SQL('''
            DELETE FROM {id_pred}
            WHERE cnnstate IN %s;
            DELETE FROM {id_cnnstate}
            WHERE id IN %s;
        ''').format(
                id_pred=sql.Identifier(project, 'prediction'),
                id_cnnstate=sql.Identifier(project, 'cnnstate')
            ),
            (tuple(uuids),tuple(uuids))
        )
        return modelIDs_invalid


    
    def get_model_training_statistics(self, project, modelStateIDs=None, modelLibraries=None):
        '''
            Assembles statistics as returned by the model (if done so). Returned
            statistics may be dicts with keys for variable names and values for
            each model state. None is appended for each model state and variable
            that does not exist.
            The optional input "modelStateIDs" may be a str, UUID, or list of str
            or UUID, and indicates a filter for model state IDs to be included
            in the assembly.
            Optional input "modelLibraries" may be a str or list of str and filters
            model libraries that were used in the project over time.
        '''
        # verify IDs
        if modelStateIDs is not None:
            if not isinstance(modelStateIDs, Iterable):
                modelStateIDs = [modelStateIDs]
            uuids = []
            for m in modelStateIDs:
                try:
                    uuids.append((UUID(m),))
                except:
                    pass
            modelStateIDs = uuids
            if not len(modelStateIDs):
                modelStateIDs = None
        
        sqlFilter = ''
        queryArgs = []

        # verify libraries
        if modelLibraries is not None:
            if not isinstance(modelLibraries, Iterable):
                modelLibraries = [modelLibraries]
            if len(modelLibraries):
                modelLibraries = tuple([(str(m),) for m in modelLibraries])
                queryArgs.append((modelLibraries,))
                sqlFilter = 'WHERE model_library IN %s'
        
        # get all statistics
        if modelStateIDs is not None and len(modelStateIDs):
            if len(sqlFilter):
                sqlFilter += ' AND id IN %s'
            else:
                sqlFilter = 'WHERE id IN %s'
            queryArgs.append((tuple([(m,) for m in modelStateIDs]),))

        queryResult = self.dbConn.execute(sql.SQL('''
            SELECT id, model_library, EXTRACT(epoch FROM timeCreated) AS timeCreated, stats FROM {id_cnnstate}
            {sql_filter}
            ORDER BY timeCreated ASC;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate'),
            sql_filter=sql.SQL(sqlFilter)
        ), (queryArgs if len(queryArgs) else None), 'all')

        # assemble output stats
        if queryResult is None or not len(queryResult):
            return {}
        else:
            # separate outputs for each model library used
            modelLibs = set([q['model_library'] for q in queryResult])
            ids, dates, stats_raw = dict((m, []) for m in modelLibs), dict((m, []) for m in modelLibs), dict((m, []) for m in modelLibs)
            keys = dict((m, set()) for m in modelLibs)

            for q in queryResult:
                modelLib = q['model_library']
                ids[modelLib].append(str(q['id']))
                dates[modelLib].append(q['timecreated'])
                try:
                    qDict = json.loads(q['stats'])
                    stats_raw[modelLib].append(qDict)
                    keys[modelLib] = keys[modelLib].union(set(qDict.keys()))
                except:
                    stats_raw[modelLib].append({})
                    
            # assemble into series
            series = {}
            for m in modelLibs:
                series[m] = dict((k, len(stats_raw[m])*[None]) for k in keys[m])

                for idx, stat in enumerate(stats_raw[m]):
                    for key in keys[m]:
                        if key in stat:
                            series[m][key][idx] = stat[key]

            return {
                'ids': ids,
                'timestamps': dates,
                'series': series
            }