'''
    Middleware for AIController: handles requests and updates to and from the database.

    2019-20 Benjamin Kellenberger
'''

from datetime import datetime
from modules.AIController.backend import celery_interface as aic_int
from modules.AIWorker.backend import celery_interface as aiw_int
import celery
from celery import current_app, group
from kombu import Queue
from psycopg2 import sql
from util.helpers import current_time
from .messageProcessor import MessageProcessor
from .annotationWatchdog import Watchdog
from modules.AIController.taskWorkflow.workflowDesigner import WorkflowDesigner
from modules.Database.app import Database
from util.helpers import array_split

from .sql_string_builder import SQLStringBuilder


class AIMiddleware():

    def __init__(self, config, passiveMode=False):
        self.config = config
        self.dbConn = Database(config)
        self.sqlBuilder = SQLStringBuilder(config)
        self.passiveMode = passiveMode

        #TODO: replace with messageProcessor property:
        self.training = {}   # dict of bools for each project; will be set to True once start_training is called (and False as soon as everything about the training process has finished)

        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()

        self.messageProcessor = MessageProcessor(self.celery_app)
        if not self.passiveMode:
            print("Active mode")
            self._init_project_queues()
            self.watchdogs = {}    # one watchdog per project. Note: watchdog only created if users poll status (i.e., if there's activity)
            self.workflowDesigner = WorkflowDesigner(self.dbConn, self.celery_app)
            self.messageProcessor.start()

    
    def _init_project_queues(self):
        '''
            Queries the database for projects that support AIWorkers
            and adds respective management queues to Celery to listen to them.
        '''
        #TODO: deprecated?
        return
        if self.passiveMode:
            return
        if current_app.conf.task_queues is not None:
            queues = list(current_app.conf.task_queues)
            current_queue_names = set([c.name for c in queues])
            # print('Existing queues: {}'.format(', '.join([c.name for c in queues])))
        else:
            queues = []
            current_queue_names = set()
        
        log_updates = []
        projects = self.dbConn.execute('SELECT shortname FROM aide_admin.project WHERE ai_model_enabled = TRUE;', None, 'all')
        if len(projects):
            for project in projects:
                pName = project['shortname'] + '_aic'
                if not pName in current_queue_names:
                    current_queue_names.add(pName)
                    queues.append(Queue(pName))
                    current_app.control.add_consumer(pName)     #TODO: stalls if Celery app is not set up. Might be ok to just skip this
                    log_updates.append(pName)
            
            current_app.conf.update(task_queues=tuple(queues))
            if len(log_updates):
                print('Added queue(s) for project(s): {}'.format(', '.join(log_updates)))


    def _init_watchdog(self, project, nudge=False):
        '''
            Launches a thread that periodically polls the database for new
            annotations. Once the required number of new annotations is reached,
            this thread will initiate the training process through the middleware.
            The thread will be terminated and destroyed; a new thread will only be
            re-created once the training process has finished.
        '''
        if self.passiveMode:
            return
        if project in self.watchdogs and self.watchdogs[project] == False:
                # project does not need a watchdog
                #TODO: implement force re-check after e.g. settings update
                return
        
        else:
            # check if auto-training enabled
            projSettings = self.dbConn.execute('''
                SELECT ai_model_enabled, numImages_autoTrain
                FROM aide_admin.project
                WHERE shortname = %s
                ''', (project,), 1)

            if projSettings is None or not len(projSettings) or \
                not (projSettings[0]['ai_model_enabled']) or \
                    projSettings[0]['numimages_autotrain'] is None or projSettings[0]['numimages_autotrain'] <= 0:
                # no watchdog to be configured; set flag to avoid excessive DB queries
                #TODO: enable on-the-fly project settings updates to this end
                self.watchdogs[project] = False
                return

        # init watchdog            
        self.watchdogs[project] = Watchdog(project, self.config, self.dbConn, self)
        self.watchdogs[project].start()

        if nudge:
            self.watchdogs[project].nudge()


    def _get_num_available_workers(self):
        #TODO: message + queue if no worker available
        #TODO: limit to n tasks per worker
        i = self.celery_app.control.inspect()
        if i is not None:
            stats = i.stats()
            if stats is not None:
                return len(i.stats())
        return 1    #TODO

    def _get_project_settings(self, project):
        queryStr = sql.SQL('''SELECT numImages_autoTrain,
            minNumAnnoPerImage, maxNumImages_train,maxNumImages_inference
            FROM aide_admin.project WHERE shortname = %s;''')
        settings = self.dbConn.execute(queryStr, (project,), 1)[0]
        return settings


    def _get_training_job_signature(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
        '''
            Assembles (but does not submit) a training job based on the provided parameters.
        '''
        # check if training is still in progress
        if self.messageProcessor.task_ongoing(project, ('AIWorker.call_train', 'AIWorker.call_average_model_states')):
            raise Exception('Training process already running.')

        self.training[project] = True


        try:
            # sanity checks
            if not (isinstance(minTimestamp, datetime) or minTimestamp == 'lastState' or
                    minTimestamp == -1 or minTimestamp is None):
                raise ValueError('{} is not a recognized property for variable "minTimestamp"'.format(str(minTimestamp)))

            # identify number of available workers
            if maxNumWorkers != 1:
                # only query the number of available workers if more than one is specified to save time
                num_workers = min(maxNumWorkers, self._get_num_available_workers())
            else:
                num_workers = maxNumWorkers

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

            if maxNumImages is None:
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

            if maxNumWorkers > 1:

                # distribute across workers (TODO: also specify subset size for multiple jobs; randomly draw if needed)
                images_subset = array_split(imageIDs, max(1, len(imageIDs) // num_workers))

                processes = []
                for subset in images_subset:
                    processes.append(celery_interface.call_train.si(project, subset, True))
                process = group(processes)

            else:
                # call one worker directly
                process = celery_interface.call_train.si(project, imageIDs, False)
            
            return process, num_workers

        except:
            self.training = self.messageProcessor.task_ongoing(project, 'train')
            return None


    def _get_inference_job_signature(self, project, imageIDs, maxNumWorkers=-1):
        '''
            Assembles (but does not submit) an inference job based on the provided parameters.
        '''
        # setup
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_available = self._get_num_available_workers()
            if maxNumWorkers == -1:
                maxNumWorkers = num_available   #TODO: more than one process per worker?
            else:
                maxNumWorkers = min(maxNumWorkers, num_available)

        # distribute across workers
        images_subset = array_split(imageIDs, max(1, len(imageIDs) // maxNumWorkers))
        jobs = []
        for subset in images_subset:
            job = aiw_int.call_inference.si(project=project, imageIDs=subset)
            jobs.append(job)

        jobGroup = group(jobs)
        return jobGroup


    def task_ongoing(self, project, taskTypes):
        '''
            Polls Celery via MessageProcessor for a given list of
            task types (or a single task type string) and returns
            True if at least one of the taskTypes provided is cur-
            rently running.
        '''
        return self.messageProcessor.task_ongoing(project, taskTypes)


    def aide_internal_notify(self, message):
        '''
            Used for AIDE administrative communication between AIController
            and AIWorker(s), e.g. for setting up queues.
        '''
        if self.passiveMode:
            return
        if 'task' in message:
            if message['task'] == 'add_projects':
                self._init_project_queues()


    def get_training_images(self, project, minTimestamp='lastState', includeGoldenQuestions=True,
                            minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
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

        # identify number of available workers
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_workers = min(maxNumWorkers, self._get_num_available_workers())
        else:
            num_workers = maxNumWorkers

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

        if maxNumImages is None:
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

        if maxNumWorkers > 1:
            # split for distribution across workers (TODO: also specify subset size for multiple jobs; randomly draw if needed)
            imageIDs = array_split(imageIDs, max(1, len(imageIDs) // num_workers))
        else:
            imageIDs = [imageIDs]

        return imageIDs


    def get_inference_images(self, project, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, maxNumWorkers=-1):
        '''
            Queries the database for the latest images to be used for inference after model training.
            Returns a list with image UUIDs accordingly, split into the number of available workers.
            #TODO: goldenQuestionsOnly
        '''
        if maxNumImages is None or maxNumImages == -1:
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

        # split for distribution across workers
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_available = self._get_num_available_workers()
            if maxNumWorkers == -1:
                maxNumWorkers = num_available   #TODO: more than one process per worker?
            else:
                maxNumWorkers = min(maxNumWorkers, num_available)
        
        if maxNumWorkers > 1:
            imageIDs = array_split(imageIDs, max(1, len(imageIDs) // maxNumWorkers))
        else:
            imageIDs = [imageIDs]
        return imageIDs

    
    def start_training(self, project, numEpochs=1, minTimestamp='lastState', includeGoldenQuestions=True, minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
        '''
            Initiates a training round for the model, based on the set of data (images, annotations)
            as specified in the parameters. Distributes data to the set of registered AIWorker instan-
            ces, which then call the 'train' function of the AI model given in the configuration. Upon
            training completion, the model states as returned by the function, and eventually the 
            AIWorker instances are collected, and the AIController calls the 'average_states' function
            of the AI model to get one single, most recent state. This is finally inserted to the data-
            base.
            Note that this function only loads the annotation data from the database, but not the images.
            Retrieving images is part of the AI model's 'train' function. TODO: feature vectors?

            Input parameters:
            - project: The project to perform training on
            - minTimestamp: Defines the earliest point in time of the annotations to be considered for
                            model training. May take one of the following values:
                            - 'lastState' (default): Limits the annotations to those made after the time-
                                                     stamp of the latest model state. If no model state is
                                                     found, all annotations are considered.
                            - None, -1, or 'all': Includes all annotations.
                            - (a datetime object), int or float: Includes annotations made after a custom
                                                                 timestamp.
            - minNumAnnoPerImage: Minimum number of annotations per image to be considered for training.
                                  This may be useful for e.g. detection tasks with a lot of false alarms
                                  in order to limit the "forgetting factor" of the model subject to training.
            - maxNumImages: Maximum number of images to train on at a time.
            - maxNumWorkers: Specify the maximum number of workers to distribute training to. If set to 1,
                             the model is trained on just one worker (no model state averaging appended).
                             If set to a number, that number of workers (up to the maximum number of connected)
                             is consulted for training the model. Upon completion, all model state dictionaries
                             are averaged by one random worker.
                             If set to -1, all connected workers are considered. //TODO: load balancing?

            Returns:
            - A dict with a status message. May take one of the following:
                - TODO: status ok, fail, no annotations, etc. Make threaded so that it immediately returns something.
        '''
        # identify number of available workers  #TODO: this fixes the number of workers at the start, even for multiple epochs...
        if maxNumWorkers < 0:
            numWorkers = min(max(1, maxNumWorkers), self._get_num_available_workers())
        elif maxNumWorkers > 1:
            numWorkers = min(maxNumWorkers, self._get_num_available_workers())
        else:
            numWorkers = maxNumWorkers
        numEpochs = max(1, numEpochs)
        
        def _get_training_signature(epoch=1):
            return celery.chain(aic_int.get_training_images.s(**{'project':project,
                                                    'epoch':epoch,
                                                    'minTimestamp':minTimestamp,
                                                    'includeGoldenQuestions':includeGoldenQuestions,
                                                    'minNumAnnoPerImage':minNumAnnoPerImage,
                                                    'maxNumImages':maxNumImages,
                                                    'numWorkers':numWorkers}).set(queue='AIController'),
                                celery.chord(
                                    [aiw_int.call_train.s(**{'index':i, 'epoch':epoch, 'project':project}).set(queue='AIWorker') for i in range(numWorkers)],
                                    aiw_int.call_average_model_states.si(**{'epoch':epoch, 'project':project}).set(queue='AIWorker')
                                )
                    )

        process = celery.chain(_get_training_signature(e) for e in range(numEpochs))

        # submit job
        task_id = self.messageProcessor.task_id(project)
        job = process.apply_async(task_id=task_id,
                        # ignore_result=False,
                        # result_extended=True,
                        headers={'headers':{'project':project,'type':'train','submitted': str(current_time())}})

        # start listener
        self.messageProcessor.register_job(project, job, 'train')
        print("Completed.")
        return 'ok'


    def _do_inference(self, project, process):

        # send job
        task_id = self.messageProcessor.task_id(project)
        result = process.apply_async(task_id=task_id,
                        queue='AIWorker',
                        # ignore_result=False,
                        # result_extended=True,
                        headers={'headers':{'project':project,'type':'inference','submitted': str(current_time())}})

        # start listener
        self.messageProcessor.register_job(project, result, 'inference')

        return


    def start_inference(self, project, forceUnlabeled=True, maxNumImages=-1, maxNumWorkers=-1):
        '''
            Performs inference (prediction) on a set of data (images) as specified in the parameters. Distributes
            data to the set of registered AIWorker instances, which then call the 'inference' function of the AI
            model given in the configuration. Upon completion, each AIWorker then automatically inserts the latest
            predictions into the database and reports back to the AIController (this instance) that its job has
            finished.
            Note that this function only loads the annotation data from the database, but not the images.
            Retrieving images is part of the AI model's 'train' function.
            The AI model, depending on its configuration, may or may not choose to load the images, but just work
            with the feature vectors as provided through the database directly. This is particularly useful for mo-
            dels that are supposed to have e.g. a frozen feature extractor, but fine-tune the last prediction branch
            at each inference time to accelerate the process.

            Input parameters:
            - project: The project to perform inference in
            - forceUnlabeled: If True, only images that have not been labeled (i.e., with a viewcount of 0) will be
                              predicted on (default).
            - maxNumImages: Manually override the project settings' maximum number of images to do inference on.
                            If set to -1 (default), the value from the project settings will be chosen.
            - maxNumWorkers: Manually set the maximum number of AIWorker instances to perform inference at the same
                             time. If set to -1 (default), the data will be divided across all registered workers.
        '''

        # identify number of available workers  #TODO: this fixes the number of workers at the start, even for multiple epochs...
        if maxNumWorkers < 0:
            numWorkers = min(max(1, maxNumWorkers), self._get_num_available_workers())
        elif maxNumWorkers > 1:
            numWorkers = min(maxNumWorkers, self._get_num_available_workers())
        else:
            numWorkers = maxNumWorkers
        def _get_inference_signature():
            return celery.chain(aic_int.get_inference_images.s(**{'project':project,
                                                    'goldenQuestionsOnly':False,        #TODO
                                                    'forceUnlabeled':forceUnlabeled,
                                                    'maxNumImages':maxNumImages,
                                                    'numWorkers':numWorkers}).set(queue='AIController'),
                                celery.group(
                                    [aiw_int.call_inference.s(**{'index':i, 'epoch':None, 'project':project}).set(queue='AIWorker') for i in range(numWorkers)],
                                )
                    )
        
        process = _get_inference_signature()

        # submit job
        task_id = self.messageProcessor.task_id(project)
        job = process.apply_async(task_id=task_id,
                        # ignore_result=False,
                        # result_extended=True,
                        headers={'headers':{'project':project,'type':'inference','submitted': str(current_time())}})

        # start listener
        self.messageProcessor.register_job(project, job, 'inference')
        print("Completed.")
        return 'ok'

        # # setup
        # if maxNumImages is None or maxNumImages == -1:
        #     queryResult = self.dbConn.execute('''
        #         SELECT maxNumImages_inference
        #         FROM aide_admin.project
        #         WHERE shortname = %s;''', (project,), 1)
        #     maxNumImages = queryResult['maxnumimages_inference']            
        # queryVals = (maxNumImages,)

        # # load the IDs of the images that are being subjected to inference
        # sql = self.sqlBuilder.getInferenceQueryString(project, forceUnlabeled, maxNumImages)
        # imageIDs = self.dbConn.execute(sql, queryVals, 'all')
        # imageIDs = [i['image'] for i in imageIDs]

        # process = self._get_inference_job_signature(project, imageIDs, maxNumWorkers)
        # self._do_inference(project, process)
        # return 'ok'


    
    def inference_fixed(self, project, imageIDs, maxNumWorkers=-1):
        '''
            Performs inference (prediction) on a fixed set of data (images), as provided by the parameter 'imageIDs'.
            Distributes data to the set of registered AIWorker instances, which then call the 'inference' function of
            the AI model given in the configuration. Upon completion, each AIWorker then automatically inserts the
            latest predictions into the database and reports back to the AIController (this instance) that its job has
            finished.
            Note that this function only loads the annotation data from the database, but not the images.
            Retrieving images is part of the AI model's 'train' function.
            The AI model, depending on its configuration, may or may not choose to load the images, but just work
            with the feature vectors as provided through the database directly. This is particularly useful for mo-
            dels that are supposed to have e.g. a frozen feature extractor, but fine-tune the last prediction branch
            at each inference time to accelerate the process.

            Input parameters:
            - project: The project to perform inference in
            - imageIDs: An array containing the UUIDs (or equivalent strings) of the images that need to be inferred on.
            - maxNumWorkers: Manually set the maximum number of AIWorker instances to perform inference at the same
                             time. If set to -1 (default), the data will be divided across all registered workers.
        '''

        process = self._get_inference_job_signature(project, imageIDs, maxNumWorkers)
        self._do_inference(project, process)
        return 'ok'
    


    def start_train_and_inference(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=None, 
                                    maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=None, maxNumWorkers_inference=1):
        '''
            Submits a model training job, followed by inference.
            This is the default behavior for the automated model update, since the newly trained model should directly
            be used to infer new, potentially useful labels.
        '''
        #TODO: replace with customizable train-predict chains

        # get training job signature
        process, numWorkers_train = self._get_training_job_signature(
                                        project=project,
                                        minTimestamp=minTimestamp,
                                        minNumAnnoPerImage=minNumAnnoPerImage,
                                        maxNumImages=maxNumImages_train,
                                        maxNumWorkers=maxNumWorkers_train)

        # submit job
        task_id = self.messageProcessor.task_id(project)
        if numWorkers_train > 1:
            # also append average model states job
            job = process.apply_async(task_id=task_id,
                            queue='AIWorker',
                            # ignore_result=False,
                            # result_extended=True,
                            headers={'headers':{'project':project,'type':'train','submitted': str(current_time())}},
                            link=aiw_int.call_average_model_states.s(1, project))      #TODO: epoch
        else:
            job = process.apply_async(task_id=task_id,
                            queue='AIWorker',
                            # ignore_result=False,
                            # result_extended=True,
                            headers={'headers':{'project':project,'type':'train','submitted': str(current_time())}})
        

        # start listener thread
        # def chain_inference(*args):
        #     self.training = self.messageProcessor.task_ongoing(project, 'train')
        #     return self.start_inference(project, forceUnlabeled_inference, maxNumImages_inference, maxNumWorkers_inference)

        self.messageProcessor.register_job(project, job, 'train')

        return 'ok'


    
    def launch_task(self, project, workflow):
        '''
            Accepts a dict containing a workflow definition as per workflow designer,
            parses it and launches the job if valid.
            Returns the task ID (TODO: useless due to sub-IDs being generated).
        '''
        process = self.workflowDesigner.parseWorkflow(project, workflow)

        task_id = self.messageProcessor.task_id(project)
        job = process.apply_async(task_id=task_id,
                            queue='AIWorker',
                            ignore_result=False,
                            # result_extended=True,
                            headers={'headers':{'project':project,'submitted': str(current_time())}})
        
        self.messageProcessor.register_job(project, job, 'workflow')    #TODO: task type; task ID for polling...

        return task_id



    def check_status(self, project, checkProject, checkTasks, checkWorkers):
        '''
            Queries the Celery worker results depending on the parameters specified.
            Returns their status accordingly if they exist.
        '''
        status = {}

        # project status
        if checkProject:
            if self.training:
                status['project'] = {}
            else:
                # notify watchdog that users are active
                if not project in self.watchdogs or \
                    (self.watchdogs[project] != False and self.watchdogs[project].stopped()):
                    self._init_watchdog(project, True)
                if self.watchdogs[project] != False:
                    status['project'] = {
                        'num_annotated': self.watchdogs[project].lastCount,
                        'num_next_training': self.watchdogs[project].getThreshold()
                    }
                else:
                    status['project'] = {}


        # running tasks status
        if checkTasks:
            status['tasks'] = self.messageProcessor.poll_status(project)

        # get worker status (this is very expensive, as each worker needs to be pinged)
        if checkWorkers:
            status['workers'] = self.messageProcessor.poll_worker_status(project)
        
        return status



    def getAvailableAImodels(self):
        # import available modules
        from ai import PREDICTION_MODELS, ALCRITERION_MODELS
        return {
            'models': {
                'prediction': PREDICTION_MODELS,
                'ranking': ALCRITERION_MODELS
            }
        }



    def listModelStates(self, project):
        modelLibraries = self.getAvailableAImodels()
        queryStr = sql.SQL('''
            SELECT id, EXTRACT(epoch FROM timeCreated) AS time_created, model_library, alCriterion_library, num_pred
            FROM {id_cnnstate} AS cnnstate
            LEFT OUTER JOIN (
                SELECT cnnstate, COUNT(cnnstate) AS num_pred
                FROM {id_pred}
                GROUP BY cnnstate
            ) AS pred
            ON cnnstate.id = pred.cnnstate
            ORDER BY time_created DESC;
        ''').format(
            id_cnnstate=sql.Identifier(project, 'cnnstate'),
            id_pred=sql.Identifier(project, 'prediction')
        )
        result = self.dbConn.execute(queryStr, None, 'all')
        response = []
        for r in result:
            try:
                modelLibrary = modelLibraries['models']['prediction'][r['model_library']]
            except:
                modelLibrary = {
                    'name': '(not found)'
                }
            modelLibrary['id'] = r['model_library']
            try:
                alCriterionLibrary = modelLibraries['models']['ranking'][r['alcriterion_library']]
            except:
                alCriterionLibrary = {
                    'name': '(not found)'
                }
            alCriterionLibrary['id'] = r['alcriterion_library']
            response.append({
                'id': str(r['id']),
                'time_created': r['time_created'],
                'model_library': modelLibrary,
                'al_criterion_library': alCriterionLibrary,
                'num_pred': (r['num_pred'] if r['num_pred'] is not None else 0)
            })
        return response



    def getProjectModelSettings(self, project):
        '''
            Returns the AI and AL model properties for the given project,
            as stored in the database.
        '''
        result = self.dbConn.execute('''SELECT ai_model_enabled,
                ai_model_library, ai_model_settings,
                ai_alcriterion_library, ai_alcriterion_settings,
                numImages_autoTrain, minNumAnnoPerImage,
                maxNumImages_train, maxNumImages_inference
                FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1)
        return result[0]



    def saveProjectModelSettings(self, project, settings):
        '''
            TODO
        '''
        pass