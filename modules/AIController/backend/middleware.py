'''
    Middleware for AIController: handles requests and updates to and from the database.

    2019-21 Benjamin Kellenberger
'''

from datetime import datetime
import uuid
import re
import json
from constants.annotationTypes import ANNOTATION_TYPES
from ai import PREDICTION_MODELS, ALCRITERION_MODELS
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
from modules.AIController.taskWorkflow.workflowTracker import WorkflowTracker
from modules.Database.app import Database
from modules.AIWorker.backend.fileserver import FileServer
from util.helpers import array_split, parse_parameters, get_class_executable

from .sql_string_builder import SQLStringBuilder


class AIMiddleware():

    def __init__(self, config, passiveMode=False):
        self.config = config
        self.dbConn = Database(config)
        self.sqlBuilder = SQLStringBuilder(config)
        self.passiveMode = passiveMode
        self.scriptPattern = re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script\.?>')
        self._init_available_ai_models()

        #TODO: replace with messageProcessor property:
        self.training = {}   # dict of bools for each project; will be set to True once start_training is called (and False as soon as everything about the training process has finished)

        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()
        
        if not self.passiveMode:
            self.messageProcessor = MessageProcessor(self.celery_app)
            self.watchdogs = {}    # one watchdog per project. Note: watchdog only created if users poll status (i.e., if there's activity)
            self.workflowDesigner = WorkflowDesigner(self.dbConn, self.celery_app)
            self.workflowTracker = WorkflowTracker(self.dbConn, self.celery_app)
            self.messageProcessor.start()


    def _init_available_ai_models(self):
        #TODO: 1. using regex to remove scripts is not failsafe; 2. ugly code...
        models = {
            'prediction': PREDICTION_MODELS,
            'ranking': ALCRITERION_MODELS
        }

        # remove script tags and validate annotation and prediction types specified
        for modelKey in models['prediction']:
            model = models['prediction'][modelKey]
            if 'name' in model and isinstance(model['name'], str):
                model['name'] = re.sub(self.scriptPattern, '(script removed)', model['name'])
            else:
                model['name'] = modelKey
            if 'description' in model and isinstance(model['description'], str):
                model['description'] = re.sub(self.scriptPattern, '(script removed)', model['description'])
            else:
                model['description'] = '(no description available)'
            if 'author' in model and isinstance(model['author'], str):
                model['author'] = re.sub(self.scriptPattern, '(script removed)', model['author'])
            else:
                model['author'] = '(unknown)'
            if not 'annotationType' in model or not 'predictionType' in model:
                # no annotation and/or no prediction type specified; remove model
                print(f'WARNING: model "{modelKey}" has no annotationType and/or no predictionType specified and is therefore ignored.')
                del models['prediction'][modelKey]
                continue
            if isinstance(model['annotationType'], str):
                model['annotationType'] = [model['annotationType']]
            for idx, type in enumerate(model['annotationType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: annotation type "{type}" not understood and ignored.')
                    del model['annotationType'][idx]
            if isinstance(model['predictionType'], str):
                model['predictionType'] = [model['predictionType']]
            for idx, type in enumerate(model['predictionType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: prediction type "{type}" not understood and ignored.')
                    del model['predictionType'][idx]
            if model['annotationType'] is None or not len(model['annotationType']):
                print(f'WARNING: no valid annotation type specified for model "{modelKey}"; ignoring...')
                del models['prediction'][modelKey]
                continue
            if model['predictionType'] is None or not len(model['predictionType']):
                print(f'WARNING: no valid prediction type specified for model "{modelKey}"; ignoring...')
                del models['prediction'][modelKey]
                continue
            # default model options
            try:
                modelClass = get_class_executable(modelKey)
                defaultOptions = modelClass.getDefaultOptions()
                model['defaultOptions'] = defaultOptions
            except:
                # no default options available; append no key to signal that there's no options
                pass
            models['prediction'][modelKey] = model
        for rankerKey in models['ranking']:
            ranker = models['ranking'][rankerKey]
            if 'name' in ranker and isinstance(ranker['name'], str):
                ranker['name'] = re.sub(self.scriptPattern, '(script removed)', ranker['name'])
            else:
                ranker['name'] = rankerKey
            if 'author' in ranker and isinstance(ranker['author'], str):
                ranker['author'] = re.sub(self.scriptPattern, '(script removed)', ranker['author'])
            else:
                ranker['author'] = '(unknown)'
            if 'description' in ranker and isinstance(ranker['description'], str):
                ranker['description'] = re.sub(self.scriptPattern, '(script removed)', ranker['description'])
            else:
                ranker['description'] = '(no description available)'
            if not 'predictionType' in ranker:
                # no prediction type specified; remove ranker
                print(f'WARNING: ranker "{rankerKey}" has no predictionType specified and is therefore ignored.')
                del models['ranking'][rankerKey]
                continue
            if isinstance(ranker['predictionType'], str):
                ranker['predictionType'] = [ranker['predictionType']]
            for idx, type in enumerate(ranker['predictionType']):
                if type not in ANNOTATION_TYPES:
                    print(f'WARNING: prediction type "{type}" not understood and ignored.')
                    del ranker['predictionType'][idx]
            if ranker['predictionType'] is None or not len(ranker['predictionType']):
                print(f'WARNING: no valid prediction type specified for ranker "{rankerKey}"; ignoring...')
                del models['ranking'][rankerKey]
                continue
            # default ranker options
            try:
                rankerClass = get_class_executable(rankerKey)
                defaultOptions = rankerClass.getDefaultOptions()
                ranker['defaultOptions'] = defaultOptions
            except:
                # no default options available; append no key to signal that there's no options
                pass
            models['ranking'][rankerKey] = ranker
        self.aiModels = models


    def _init_watchdog(self, project, nudge=False, recheckAutotrainSettings=False):
        '''
            Launches a thread that periodically polls the database for new
            annotations. Once the required number of new annotations is reached,
            this thread will initiate the training process through the middleware.
            The thread will be terminated and destroyed; a new thread will only be
            re-created once the training process has finished.
        '''
        if self.passiveMode:
            return

        if project not in self.watchdogs:
            self.watchdogs[project] = Watchdog(project, self.config, self.dbConn, self)
            self.watchdogs[project].start()   
        
        if recheckAutotrainSettings:
            # also nudges the watchdog
            self.watchdogs[project].recheckAutotrainSettings()
        
        elif nudge:
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
        #TODO: not required (yet)


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

        if maxNumImages is None or not isinstance(maxNumImages, int) or maxNumImages <= 0:
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
            maxNumImages = queryResult[0]['maxnumimages_inference']    
        
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
            return celery.chain(aic_int.get_training_images.s(**{'blank':None,
                                                    'project':project,
                                                    'epoch':epoch,
                                                    'minTimestamp':minTimestamp,
                                                    'includeGoldenQuestions':includeGoldenQuestions,
                                                    'minNumAnnoPerImage':minNumAnnoPerImage,
                                                    'maxNumImages':maxNumImages,
                                                    'numWorkers':numWorkers}).set(queue='AIController'),
                                celery.chord(
                                    [aiw_int.call_train.s(**{'index':i, 'epoch':epoch, 'project':project}).set(queue='AIWorker') for i in range(numWorkers)],
                                    aiw_int.call_average_model_states.si(**{'blank':None, 'epoch':epoch, 'project':project}).set(queue='AIWorker')
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
            return celery.chain(aic_int.get_inference_images.s(**{'blank':None,
                                                    'project':project,
                                                    'epoch':1,
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
    


    #TODO: deprecated; replace with workflow
    def start_train_and_inference(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=-1, 
                                    maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=-1, maxNumWorkers_inference=1,
                                    author=None):
        '''
            Submits a model training job, followed by inference.
            This is the default behavior for the automated model update, since the newly trained model should directly
            be used to infer new, potentially useful labels.
        '''

        #TODO: sanity checks
        workflow = {
            'project': project,
            'tasks': [
                {
                    'id': '0',
                    'type': 'train',
                    'kwargs': {
                        'min_timestamp': minTimestamp,
                        'min_anno_per_image': minNumAnnoPerImage,
                        'max_num_images': maxNumImages_train,
                        'max_num_workers': maxNumWorkers_train
                    }
                },
                {
                    'id': '1',
                    'type': 'inference',
                    'kwargs': {
                        'force_unlabeled': forceUnlabeled_inference,
                        'max_num_images': maxNumImages_inference,
                        'max_num_workers': maxNumWorkers_inference
                    }
                }
            ],
            'options': {}
        }
        process = self.workflowDesigner.parseWorkflow(project, workflow)

        
        # launch workflow
        task_id = self.workflowTracker.launchWorkflow(project, process, workflow, author)

        return {
            'status': 0,
            'task_id': task_id
        }


    
    def launch_task(self, project, workflow, author=None):
        '''
            Accepts a workflow as one of the following three variants:
            - ID (str or UUID): ID of a saved workflow in this project
            - 'default': uses the default workflow for this project
            - dict: an actual workflow as per specifications
            parses it and launches the job if valid.
            Returns the task ID accordingly.
        '''
        if isinstance(workflow, str):
            if workflow.lower() == 'default':
                # load default workflow
                queryStr = sql.SQL('''
                    SELECT workflow FROM {id_workflow}
                    WHERE id = (
                        SELECT default_workflow
                        FROM aide_admin.project
                        WHERE shortname = %s
                    );
                ''').format(
                    id_workflow=sql.Identifier(project, 'workflow')
                )
                result = self.dbConn.execute(queryStr, (project,), 1)
                if result is None or not len(result):
                    return {
                        'status': 2,
                        'message': f'Workflow with ID "{str(workflow)}" does not exist in this project'
                    }
                workflow = result[0]['workflow']
            else:
                # try first to parse workflow
                try:
                    workflow = json.loads(workflow)
                except:
                    # try to convert to UUID instead
                    try:
                        workflow = uuid.UUID(workflow)
                    except:
                        return {
                            'status': 3,
                            'message': f'"{str(workflow)}" is not a valid workflow ID'
                        }
        
        if isinstance(workflow, uuid.UUID):
            # load workflow as per UUID
            queryStr = sql.SQL('''
                    SELECT workflow FROM {id_workflow}
                    WHERE id %s;
                ''').format(
                    id_workflow=sql.Identifier(project, 'workflow')
                )
            result = self.dbConn.execute(queryStr, (project,), 1)
            if result is None or not len(result):
                return {
                    'status': 2,
                    'message': f'Workflow with ID "{str(workflow)}" does not exist in this project'
                }
            workflow = result[0]['workflow']

        # try to parse workflow
        try:
            process = self.workflowDesigner.parseWorkflow(project, workflow, False)
        except Exception as e:
            return {
                'status': 4,
                'message': f'Workflow could not be parsed (message: "{str(e)}")'
            }

        task_id = self.workflowTracker.launchWorkflow(project, process, workflow, author)

        return {
            'status': 0,
            'task_id': task_id
        }



    def revoke_task(self, project, taskID, username):
        '''
            Revokes (aborts) a task with given task ID for a given
            project, if it exists.
            Also sets an entry in the database (and notes who aborted
            the task).
        '''
        self.workflowTracker.revokeTask(username, project, taskID)



    def check_status(self, project, checkProject, checkTasks, checkWorkers, nudgeWatchdog=False, recheckAutotrainSettings=False):
        '''
            Queries the Celery worker results depending on the parameters specified.
            Returns their status accordingly if they exist.
        '''
        status = {}

        # watchdog
        self._init_watchdog(project, nudgeWatchdog, recheckAutotrainSettings)

        # project status
        if checkProject:
            status['project'] = {
                'ai_auto_training_enabled': self.watchdogs[project].getAImodelAutoTrainingEnabled(),
                'num_annotated': self.watchdogs[project].lastCount,
                'num_next_training': self.watchdogs[project].getThreshold()
            }

        # running tasks status
        if checkTasks:
            status['tasks'] = self.workflowTracker.pollAllTaskStatuses(project)       #TODO: self.messageProcessor.poll_status(project)

        # get worker status (this is very expensive, as each worker needs to be pinged)
        if checkWorkers:
            status['workers'] = self.messageProcessor.poll_worker_status()
        
        return status



    #TODO
    def pollTaskStatus(self, project, taskID):
        return self.workflowTracker.pollTaskStatus(project, taskID)



    def getAvailableAImodels(self, project=None):
        if project is None:
            return {
                'models': self.aiModels
            }
        else:
            models = {
                'prediction': {},
                'ranking': self.aiModels['ranking']
            }

            # identify models that are compatible with project's annotation and prediction type
            projImmutables = self.dbConn.execute(
                '''
                    SELECT annotationtype, predictiontype
                    FROM aide_admin.project
                    WHERE shortname = %s;
                ''',
                (project,),
                1
            )
            projImmutables = projImmutables[0]
            projAnnoType = projImmutables['annotationtype']
            projPredType = projImmutables['predictiontype']
            for key in self.aiModels['prediction'].keys():
                model = self.aiModels['prediction'][key]
                if projAnnoType in model['annotationType'] and \
                    projPredType in model['predictionType']:
                    models['prediction'][key] = model

            return {
                'models': models
            }


    def verifyAImodelOptions(self, project, modelOptions, modelLibrary=None):
        '''
            Receives a dict of model options, a model library ID (optional)
            and verifies whether the provided options are valid for the model
            or not. Uses the following strategies to this end:
            1. If the AI model implements the function "verifyOptions", and
               if that function does not return None, it is being used.
            2. Else, this temporarily instanciates a new AI model with the
               given options and checks whether any errors occur. Returns
               True if not, but appends a warning that the options could not
               reliably be verified.
        '''
        # get AI model library if not specified
        if modelLibrary is None or modelLibrary not in self.aiModels['prediction']:
            modelLib = self.dbConn.execute('''
                SELECT ai_model_library
                FROM aide_admin.project
                WHERE shortname = %s;
            ''', (project,), 1)
            modelLibrary = modelLib[0]['ai_model_library']

        modelName = self.aiModels['prediction'][modelLibrary]['name']

        response = None

        modelClass = get_class_executable(modelLibrary)
        if hasattr(modelClass, 'verifyOptions'):
            response = modelClass.verifyOptions(modelOptions)
        
        if response is None:
            # no verification implemented; alternative
            #TODO: can we always do that on the AIController?
            try:
                modelClass(project=project,
                            config=self.config,
                            dbConnector=self.dbConn,
                            fileServer=FileServer(self.config).get_secure_instance(project),
                            options=modelOptions)
                response = {
                    'valid': True,
                    'warnings': [f'A {modelName} instance could be launched, but the settings could not be verified.']
                }
            except Exception as e:
                # model could not be instantiated; append error
                response = {
                    'valid': False,
                    'errors': [ str(e) ]
                }
        
        return response


    def updateAImodelSettings(self, project, settings):
        '''
            Updates the project's AI model settings.
            Verifies whether the specified AI and ranking model libraries
            exist on this setup of AIDE. Raises an exception otherwise.

            Also tries to verify any model options provided with the
            model's built-in function (if present and implemented).
            Returns warnings, errors, etc. about that.
        '''
        # AI libraries installed in AIDE
        availableModels = self.getAvailableAImodels()['models']

        # project immutables
        projImmutables = self.dbConn.execute(
            '''
                SELECT annotationtype, predictiontype
                FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1
        )
        projImmutables = projImmutables[0]
        annoType = projImmutables['annotationtype']
        predType = projImmutables['predictiontype']

        # cross-check submitted tokens
        fieldNames = [
            ('ai_model_enabled', bool),
            ('ai_model_library', str),
            ('ai_alcriterion_library', str),
            ('numimages_autotrain', int),           #TODO: replace this and next four entries with default workflow
            ('minnumannoperimage', int),
            ('maxnumimages_train', int),
            ('maxnumimages_inference', int),
            ('segmentation_ignore_unlabeled', bool)
        ]
        settings_new, settingsKeys_new = parse_parameters(settings, fieldNames, absent_ok=True, escape=True, none_ok=True)

        # verify settings
        addBackgroundClass = False
        forceDisableAImodel = False
        for idx, key in enumerate(settingsKeys_new):
            if key == 'ai_model_library':
                modelLib = settings_new[idx]
                if modelLib is None or len(modelLib.strip()) == 0:
                    # no model library specified; disable AI model
                    forceDisableAImodel = True
                else:
                    if not modelLib in availableModels['prediction']:
                        raise Exception(f'Model library "{modelLib}" is not installed in this instance of AIDE.')
                    selectedModel = availableModels['prediction'][modelLib]
                    validAnnoTypes = ([selectedModel['annotationType']] if isinstance(selectedModel['annotationType'], str) else selectedModel['annotationType'])
                    validPredTypes = ([selectedModel['predictionType']] if isinstance(selectedModel['predictionType'], str) else selectedModel['predictionType'])
                    if not annoType in validAnnoTypes:
                        raise Exception(f'Model "{modelLib}" does not support annotations of type "{annoType}".')
                    if not predType in validPredTypes:
                        raise Exception(f'Model "{modelLib}" does not support predictions of type "{predType}".')
            
            elif key == 'ai_model_settings':
                # model settings are verified separately
                continue
                
            elif key == 'ai_alcriterion_library':
                modelLib = settings_new[idx]
                if modelLib is None or len(modelLib.strip()) == 0:
                    # no AL criterion library specified; disable AI model
                    forceDisableAImodel = True
                else:
                    if not modelLib in availableModels['ranking']:
                        raise Exception(f'Ranking library "{modelLib}" is not installed in this instance of AIDE.')
            
            elif key == 'ai_alcriterion_settings':
                # verify model parameters
                #TODO: outsource as well?
                pass

            elif key == 'segmentation_ignore_unlabeled':
                # only check if annotation type is segmentation mask
                if annoType == 'segmentationMasks' and settings_new[idx] is False:
                    # unlabeled areas are to be treated as "background": add class if not exists
                    addBackgroundClass = True

        if forceDisableAImodel:
            # switch flag
            flagFound = False
            for idx, key in enumerate(settingsKeys_new):
                if key == 'ai_model_enabled':
                    settings_new[idx] = False
                    flagFound = True
                    break
            if not flagFound:
                settings_new.append(False)
                settingsKeys_new.append('ai_model_enabled')

        # all checks passed; update database
        settings_new.append(project)
        queryStr = sql.SQL('''UPDATE aide_admin.project
            SET
            {}
            WHERE shortname = %s;
            '''
        ).format(
            sql.SQL(',').join([sql.SQL('{} = %s'.format(item)) for item in settingsKeys_new])
        )
        self.dbConn.execute(queryStr, tuple(settings_new), None)

        if addBackgroundClass:
            labelClasses = self.dbConn.execute(sql.SQL('''
                    SELECT * FROM {id_lc}
                ''').format(id_lc=sql.Identifier(project, 'labelclass')),
                None, 'all')
            hasBackground = False
            for lc in labelClasses:
                if lc['idx'] == 0:
                    hasBackground = True
                    break
            if not hasBackground:
                # find unique name
                lcNames = set([lc['name'] for lc in labelClasses])
                bgName = 'background'
                counter = 0
                while bgName in lcNames:
                    bgName = f'background ({counter})'
                    counter += 1
                self.dbConn.execute(sql.SQL('''
                    INSERT INTO {id_lc} (name, idx, hidden)
                    VALUES (%s, 0, true)
                ''').format(id_lc=sql.Identifier(project, 'labelclass')),
                (bgName,), None)

        response = {'status': 0}

        # check for and verify AI model settings
        if 'ai_model_settings' in settings:
            aiModelOptionsStatus = self.saveProjectModelSettings(project, settings['ai_model_settings'])
            response['ai_model_settings_status'] = aiModelOptionsStatus

        return response


    def listModelStates(self, project):
        modelLibraries = self.getAvailableAImodels()

        # get meta data about models shared through model marketplace
        result = self.dbConn.execute('''
            SELECT id, origin_uuid,
            author, anonymous, public,
            shared, tags, name, description
            FROM aide_admin.modelMarketplace
            WHERE origin_project = %s OR origin_project IS NULL;
        ''', (project,), 'all')
        if result is not None and len(result):
            modelMarketplaceMeta = {}
            for r in result:
                mmID = r['id']
                values = {}
                for key in r.keys():
                    if isinstance(r[key], uuid.UUID):
                        values[key] = str(r[key])
                    else:
                        values[key] = r[key]
                modelMarketplaceMeta[mmID] = values
        else:
            modelMarketplaceMeta = {}

        # get project-specific model states
        queryStr = sql.SQL('''
            SELECT id, marketplace_origin_id, EXTRACT(epoch FROM timeCreated) AS time_created, model_library, alCriterion_library, num_pred
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
        if result is not None and len(result):
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

                if r['marketplace_origin_id'] in modelMarketplaceMeta:
                    marketplaceInfo = modelMarketplaceMeta[r['marketplace_origin_id']]
                else:
                    marketplaceInfo = {}

                response.append({
                    'id': str(r['id']),
                    'time_created': r['time_created'],
                    'model_library': modelLibrary,
                    'al_criterion_library': alCriterionLibrary,
                    'num_pred': (r['num_pred'] if r['num_pred'] is not None else 0),
                    'marketplace_info': marketplaceInfo
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
        # verify settings first
        optionsVerification = self.verifyAImodelOptions(project, settings)
        if optionsVerification['valid']:
            # save
            if isinstance(settings, dict):
                settings = json.dumps(settings)
            self.dbConn.execute('''
                UPDATE aide_admin.project
                SET ai_model_settings = %s
                WHERE shortname = %s;
            ''', (settings, project), None)
        else:
            optionsVerification['errors'].append('Model options have not passed verification and where therefore not saved.')
        return optionsVerification

    

    def getSavedWorkflows(self, project):
        queryStr = sql.SQL('''
            SELECT *
            FROM {id_workflow} AS wf
            LEFT OUTER JOIN (
                SELECT default_workflow
                FROM aide_admin.project
                WHERE shortname = %s
            ) AS defwf
            ON wf.id = defwf.default_workflow;
        ''').format(id_workflow=sql.Identifier(project, 'workflow'))
        result = self.dbConn.execute(queryStr, (project,), 'all')
        response = {}
        for r in result:
            response[str(r['id'])] = {
                'name': r['name'],
                'workflow': r['workflow'],
                'author': r['username'],
                'time_created': r['timecreated'].timestamp(),
                'time_modified': r['timemodified'].timestamp(),
                'default_workflow': (True if r['default_workflow'] is not None else False)
            }
        return response


    
    def saveWorkflow(self, project, username, workflow, workflowID, workflowName, setDefault=False):
        '''
            Receives a workflow definition (Python dict) to be saved
            in the database for a given project under a provided user
            name. The workflow definition is first parsed by the
            WorkflowDesigner and checked for validity. If it passes,
            it is stored in the database. If "setDefault" is True, the
            current workflow is set as the standard workflow, to be
            used for automated model training.
            Workflows can also be updated if an ID is specified.
        '''
        try:
            # check validity of workflow
            valid = self.workflowDesigner.parseWorkflow(project, workflow, verifyOnly=True)
            if not valid:
                raise Exception('Workflow is not valid.')   #TODO: detailed error message
            workflow = json.dumps(workflow)

            updateExisting = False
            if workflowID is not None:
                # ID provided; query first if id exists
                workflowID = uuid.UUID(workflowID)
                idExists = self.dbConn.execute(
                    sql.SQL('''
                        SELECT COUNT(*)
                        FROM {id_workflow}
                        WHERE id = %s;
                    ''').format(id_workflow=sql.Identifier(project, 'workflow')),
                    (workflowID,),
                    1
                )
                if len(idExists) and idExists[0] == 1:
                    updateExisting = True

            # commit to database
            if updateExisting:
                result = self.dbConn.execute(
                    sql.SQL('''
                        UPDATE {id_workflow}
                        SET name = %s, workflow = %s
                        WHERE id = %s
                        RETURNING id;
                    ''').format(id_workflow=sql.Identifier(project, 'workflow')),
                    (workflowName, workflow, workflowID),
                    1
                )
            else:
                result = self.dbConn.execute(
                    sql.SQL('''
                        INSERT INTO {id_workflow} (name, workflow, username)
                        VALUES (%s, %s, %s)
                        RETURNING id;
                    ''').format(id_workflow=sql.Identifier(project, 'workflow')),
                    (workflowName, workflow, username),
                    1
                )
            wid = result[0]['id']

            # set as default if requested
            if setDefault:
                self.dbConn.execute(
                    '''
                        UPDATE aide_admin.project
                        SET default_workflow = %s
                        WHERE shortname = %s;
                    ''',
                    (wid, project,),
                    None
                )
            return {
                'status': 0,
                'id': str(wid)
            }
        except Exception as e:
            return {
                'status': 1,
                'message': str(e)
            }


    
    def setDefaultWorkflow(self, project, workflowID):
        '''
            Receives a str (workflow ID) and sets the associated
            workflow as default, if it exists for a given project.
        '''
        if isinstance(workflowID, str):
            workflowID = uuid.UUID(workflowID)
        if not isinstance(workflowID, uuid.UUID):
            return {
                'status': 2,
                'message': f'Provided argument "{str(workflowID)}" is not a valid workflow ID'
            }
        queryStr = sql.SQL('''
            UPDATE aide_admin.project
            SET default_workflow = (
                SELECT id FROM {id_workflow}
                WHERE id = %s
                LIMIT 1
            )
            WHERE shortname = %s
            RETURNING default_workflow;
        ''').format(
            id_workflow=sql.Identifier(project, 'workflow')
        )
        result = self.dbConn.execute(queryStr, (workflowID, project,), 1)
        if result is None or not len(result) or str(result[0]['default_workflow']) != str(workflowID):
            return {
                'status': 3,
                'message': f'Provided argument "{str(workflowID)}" is not a valid workflow ID'
            }
        else:
            return {
                'status': 0
            }


    
    def deleteWorkflow(self, project, username, workflowID):
        '''
            Receives a str or iterable of str variables under
            "workflowID" and finds and deletes them for a given
            project. Only deletes them if they were created by
            the user provided in "username", or if they are
            deleted by a super user.
        '''
        if isinstance(workflowID, str):
            workflowID = (workflowID,)
        else:
            workflowID = tuple([(w,) for w in workflowID])

        queryStr = sql.SQL('''
            DELETE FROM {id_workflow}
            WHERE username = %s
            OR username IN (
                SELECT name
                FROM aide_admin.user
                WHERE isSuperUser = true
            )
            AND id IN %s
            RETURNING id;
        ''').format(
            id_workflow=sql.Identifier(project, 'workflow')
        )
        result = self.dbConn.execute(queryStr, (username, workflowID,), 'all')
        result = [str(r['id']) for r in result]

        return {
            'status': 0,
            'workflowIDs': result
        }