'''
    Middleware for AIController: handles requests and updates to and from the database.

    2019 Benjamin Kellenberger
'''

from uuid import UUID
from datetime import datetime
import pytz
import dateutil.parser
from util.helpers import current_time
import threading        # we can use Threads since we do not need parallel execution for the listener tasks
import cgi
from modules.AIController.backend import celery_interface
import celery
from celery import current_app, group
from celery.result import AsyncResult
from modules.AIWorker.backend.worker import functional
from .messageProcessor import MessageProcessor
from .annotationWatchdog import Watchdog
from .sql_string_builder import SQLStringBuilder
from modules.Database.app import Database
from util.helpers import array_split



class AIMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConn = Database(config)
        self.sqlBuilder = SQLStringBuilder(config)

        self.training = False   # will be set to True once start_training is called (and False as soon as everything about the training process has finished)

        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()


        self.watchdog = None    # note: watchdog only created if users poll status (i.e., if there's activity)

        #TODO
        self.messageProcessor = MessageProcessor()
        self.messageProcessor.start()


    def _init_watchdog(self):
        '''
            Launches a thread that periodically polls the database for new
            annotations. Once the required number of new annotations is reached,
            this thread will initiate the training process through the middleware.
            The thread will be terminated and destroyed; a new thread will only be
            re-created once the training process has finished.
        '''
        if self.training:
            return
        
        numImages_autoTrain = self.config.getProperty('AIController', 'numImages_autoTrain', type=int, fallback=-1)
        if numImages_autoTrain == -1:
            return
            
        self.watchdog = Watchdog(self.config, self.dbConn, self)
        self.watchdog.start()


    def _get_num_available_workers(self):
        #TODO: message + queue if no worker available
        #TODO: limit to n tasks per worker
        i = self.celery_app.control.inspect()
        if i is not None:
            stats = i.stats()
            if stats is not None:
                return len(i.stats())
        return 1    #TODO

    
    # def _on_raw_message(self, job, message, on_complete=None, on_failure=None):
    #     id = message['task_id']
    #     self.messages[id]['status'] = message['status']
    #     if 'result' in message:
    #         self.messages[id]['meta'] = message['result']
    #     else:
    #         self.messages[id]['meta'] = None

    #     if message['status'] == 'SUCCESS' and on_complete is not None:
    #         on_complete(job)
    #     elif message['status'] == 'FAILURE' and on_failure is not None:
    #         on_failure(job)

    def _set_initial_message(self, task_id, type):
        self.messageProcessor.messages[task_id] = {
            'type': type,
            'submitted': str(current_time()),
            'status': celery.states.PENDING,
            'meta': {'message':'sending job to worker'}
        }


    # def _listen_status(self, job, on_complete=None):
    #     '''
    #         To be called in a thread after a has been submitted.
    #         Waits for the worker to finish and stores the messages returned as
    #         the worker is moving along.
    #     '''
    #     self.messageProcessor.register_job(job, on_complete)


    def _training_completed(self, trainingJob):
        '''
            To be called after a training process has been completed.
            If there is more than one worker, the 'average_model_states'
            instruction of the AI model is called and again awaited for.
            After successful training, the 'training' flag will be set
            to False to allow another round of model training.
        '''

        # re-enable training
        self.training = False



    def _get_training_job_signature(self, minTimestamp='lastState', maxNumWorkers=-1):
        '''
            Assembles (but does not submit) a training job based on the provided parameters.
        '''
        # check if training is still in progress
        if self.training:
            raise Exception('Training process already running.')

        self.training = True

        # sanity checks
        if not (isinstance(minTimestamp, datetime) or minTimestamp == 'lastState' or
                minTimestamp == -1 or minTimestamp is None):
            raise ValueError('{} is not a recognized property for variable "minTimestamp"'.format(str(minTimestamp)))


        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            num_workers = min(maxNumWorkers, self._get_num_available_workers())
        else:
            num_workers = maxNumWorkers


        # query image IDs
        sql = self.sqlBuilder.getLatestQueryString(limit=None)

        if isinstance(minTimestamp, datetime):
            imageIDs = self.dbConn.execute(sql, (minTimestamp,), 'all')
        else:
            imageIDs = self.dbConn.execute(sql, None, 'all')

        imageIDs = [i['image'] for i in imageIDs]


        if maxNumWorkers > 1:

            # distribute across workers (TODO: also specify subset size for multiple jobs; randomly draw if needed)
            images_subset = array_split(imageIDs, max(1, len(imageIDs) // num_workers))

            processes = []
            for subset in images_subset:
                processes.append(celery_interface.call_train.si(subset, True))
            process = group(processes)

        else:
            # call one worker directly
            # process = celery_interface.call_train.delay(data) #TODO: route to specific worker? http://docs.celeryproject.org/en/latest/userguide/routing.html#manual-routing
            process = celery_interface.call_train.si(imageIDs, False)
        
        return process, num_workers


    def _get_inference_job_signature(self, imageIDs, maxNumWorkers=-1):
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
            job = celery_interface.call_inference.si(imageIDs=subset)
            jobs.append(job)

        jobGroup = group(jobs)
        return jobGroup

    
    def start_training(self, minTimestamp='lastState', maxNumWorkers=-1):
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
            - minTimestamp: Defines the earliest point in time of the annotations to be considered for
                            model training. May take one of the following values:
                            - 'lastState' (default): Limits the annotations to those made after the time-
                                                     stamp of the latest model state. If no model state is
                                                     found, all annotations are considered.
                            - None, -1, or 'all': Includes all annotations.
                            - (a datetime object): Includes annotations made after a custom timestamp.
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

        process, numWorkers = self._get_training_job_signature(minTimestamp, maxNumWorkers)


        # submit job
        task_id = self.messageProcessor.task_id()
        if numWorkers > 1:
            # also append average model states job
            job = process.apply_async(task_id=task_id, ignore_result=False, result_extended=True, link=celery_interface.call_average_model_states.s())
        else:
            job = process.apply_async(task_id=task_id, ignore_result=False, result_extended=True)

        self._set_initial_message(task_id, 'train')

        # start listener
        self.messageProcessor.register_job(job, self._training_completed)
        # t = threading.Thread(target=self._listen_status, args=(job, self._training_completed, self._training_completed))
        # t.start()

        return 'ok'


    def _do_inference(self, process):

        # send job
        task_id = self.messageProcessor.task_id()
        result = process.apply_async(task_id=task_id, ignore_result=False, result_extended=True)

        # self._set_initial_message(task_id, 'inference') #TODO: needed?

        # since inference is always done in a group, we need to create individual messages
        for child in result.children:
            self._set_initial_message(child.task_id, 'inference')

        # start listener
        self.messageProcessor.register_job(result, None)
        # t = threading.Thread(target=self._listen_status, args=(result, None, None))
        # t.start()
        return


    def start_inference(self, forceUnlabeled=True, maxNumImages=-1, maxNumWorkers=-1):
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
            - forceUnlabeled: If True, only images that have not been labeled (i.e., with a viewcount of 0) will be
                              predicted on (default).
            - maxNumImages: Manually override the project settings' maximum number of images to do inference on.
                            If set to -1 (default), the value from the project settings will be chosen.
            - maxNumWorkers: Manually set the maximum number of AIWorker instances to perform inference at the same
                             time. If set to -1 (default), the data will be divided across all registered workers.
        '''
        
        # setup
        if maxNumImages is None or maxNumImages == -1:
            maxNumImages = self.config.getProperty('AIController', 'maxNumImages_inference')

        # load the IDs of the images that are being subjected to inference
        sql = self.sqlBuilder.getInferenceQueryString(forceUnlabeled, maxNumImages)
        imageIDs = self.dbConn.execute(sql, None, 'all')
        imageIDs = [i['image'] for i in imageIDs]

        process = self._get_inference_job_signature(imageIDs, maxNumWorkers)
        self._do_inference(process)
        return 'ok'


    
    def inference_fixed(self, imageIDs, maxNumWorkers=-1):
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
            - imageIDs: An array containing the UUIDs (or equivalent strings) of the images that need to be inferred on.
            - maxNumWorkers: Manually set the maximum number of AIWorker instances to perform inference at the same
                             time. If set to -1 (default), the data will be divided across all registered workers.
        '''

        process = self._get_inference_job_signature(imageIDs, maxNumWorkers)
        self._do_inference(process)
        return 'ok'
    


    def start_train_and_inference(self, minTimestamp='lastState', maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=None, maxNumWorkers_inference=1):
        '''
            Submits a model training job, followed by inference.
            This is the default behavior for the automated model update, since the newly trained model should directly
            be used to infer new, potentially useful labels.
        '''

        # get training job signature
        process, numWorkers_train = self._get_training_job_signature(minTimestamp, maxNumWorkers_train)

        # submit job
        task_id = self.messageProcessor.task_id()
        if numWorkers_train > 1:
            # also append average model states job
            job = process.apply_async(task_id=task_id, ignore_result=False, result_extended=True, link=celery_interface.call_average_model_states.s())
        else:
            job = process.apply_async(task_id=task_id, ignore_result=False, result_extended=True)
        
        self._set_initial_message(task_id, 'train')

        # start listener thread     NOTE: we send a callback to perform inference on the latest set of images here
        def chain_inference(*args):
            self.training = False
            #TODO
            return self.start_inference(forceUnlabeled_inference, maxNumImages_inference, maxNumWorkers_inference)


        self.messageProcessor.register_job(job, chain_inference)
        # #TODO
        # t = threading.Thread(target=self._listen_status, args=(job, chain_inference, self._training_completed))
        # t.start()

        return 'ok'




    def check_status(self, project, tasks, workers):
        '''
            Queries the Celery worker results depending on the parameters specified.
            Returns their status accordingly if they exist.
        '''
        status = {}


        # project status
        if project:
            if self.training:
                status['project'] = {}
            else:
                # notify watchdog that users are active
                if self.watchdog is None or self.watchdog.stopped():
                    self._init_watchdog()
                if self.watchdog is not None:
                    self.watchdog.nudge()
                    status['project'] = {
                        'num_annotated': self.watchdog.lastCount,
                        'num_next_training': self.watchdog.annoThreshold
                    }
                else:
                    status['project'] = {}


        # running tasks status
        if tasks:
            status['tasks'] = {}
            for key in self.messageProcessor.messages.keys():
                msg = self.messageProcessor.messages[key]
                if not len(msg):
                    continue

                # check for worker failures
                if msg['status'] == celery.states.FAILURE:
                    # append failure message
                    if 'meta' in msg and isinstance(msg['meta'], BaseException):
                        info = { 'message': cgi.escape(str(msg['meta']))}
                    else:
                        info = { 'message': 'an unknown error occurred'}
                else:
                    info = msg['meta']    #TODO
                
                status['tasks'][key] = {
                    'type': msg['type'],
                    'submitted': msg['submitted'],
                    'status': msg['status'],
                    'meta': info
                }

        # get worker status (this is very expensive, as each worker needs to be pinged)
        if workers:
            workerStatus = {}
            i = self.celery_app.control.inspect()
            stats = i.stats()
            if stats is not None and len(stats):
                active_tasks = i.active()
                scheduled_tasks = i.scheduled()
                for key in stats:
                    activeTasks = [t['id'] for t in active_tasks[key]]
                    workerStatus[key] = {
                        'active_tasks': activeTasks,
                        'scheduled_tasks': scheduled_tasks[key]
                    }
            status['workers'] = workerStatus
        
        return status