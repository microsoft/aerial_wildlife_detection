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

        self.messages = {}

        self.watchdog = None    # note: watchdog only created if users poll status (i.e., if there's activity)


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
        
        numImages_autoTrain = self.config.getProperty('AIController', 'numImages_autoTrain', -1)
        if numImages_autoTrain == -1:
            return
            
        self.watchdog = Watchdog(self.config, self.dbConn, self)
        self.watchdog.start()


    
    def _on_raw_message(self, id, message):
        self.messages[id]['status'] = message['status']
        if 'result' in message:
            self.messages[id]['meta'] = message['result']
        else:
            self.messages[id]['meta'] = None



    def _inference_initiated(self, job):
        '''
            To be called in a thread after an inference process has been started.
            Waits for the worker to finish and stores the messages returned as
            the worker is moving along.
            Removes the worker's ID once finished. TODO
        '''
        job.get(on_message=lambda msg: self._on_raw_message(job.id, msg), propagate=False)



    def _training_initiated(self, jobs, distributedTraining):
        '''
            To be called in a  thread after a training process has been started.
            Waits for the worker(s) to finish and stores the messages returned
            as the worker(s) is/are moving along.
            If there is just one worker, the function simply waits for it to
            finish.
            If there is more than one worker, the 'average_model_states'
            instruction of the AI model is called and again awaited for.
            After successful training, the 'training' flag will be set
            to False to allow another round of model training.
        '''
        
        # wait for worker(s) to finish  (TODO: check if group?)
        jobs.get(on_message=lambda msg: self._on_raw_message(jobs.id, msg), propagate=False)


        # start model state averaging   (TODO: check if group?)
        job = celery_interface.call_average_model_states.si()
        statusFrame = job.freeze()
        self.messages[statusFrame.id] = {
            'type': 'modelFusion',
            'submitted': str(current_time()),
            'status': celery.states.PENDING,
            'meta': {'message':'sending job to worker'}
        }

        worker = job.delay()
        worker.get(on_message=lambda msg: self._on_raw_message(worker.id, msg), propagate=False)


        # all done, enable training again (TODO: on error/on success?)
        self.training = False

        return

    
    def start_training(self, minTimestamp='lastState', distributeTraining=False):
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
            - distributeTraining: if True, the data (images, annotations) will be split into equal chunks
                                  and distributed to all the registered AIWorkers. This requires a working
                                  implementation of the 'average_states' function within the AI model.
                                  By default (False), only one AIWorker is considered. //TODO: load balancing?

            Returns:
            - A dict with a status message. May take one of the following:
                - TODO: status ok, fail, no annotations, etc. Make threaded so that it immediately returns something.
        '''

        # check if training is still in progress
        if self.training:
            raise Exception('Training process already running.')

        self.training = True

        # sanity checks
        if not (isinstance(minTimestamp, datetime) or minTimestamp == 'lastState' or
                minTimestamp == -1 or minTimestamp is None):
            raise ValueError('{} is not a recognized property for variable "minTimestamp"'.format(str(minTimestamp)))


        # #TODO: due to a limitation of RabbitMQ, status are not being broadcast from Celery group objects.
        # # Also, it is unclear whether averaging model states from distributed trainings is useful.
        # # We therefore disable distributed training until problems are solved.
        # distributeTraining = False


        # query image IDs
        sql = self.sqlBuilder.getLatestQueryString(limit=None)

        if isinstance(minTimestamp, datetime):
            imageIDs = self.dbConn.execute(sql, (minTimestamp,), 'all')

        else:
            imageIDs = self.dbConn.execute(sql, None, 'all')

        imageIDs = [i['image'] for i in imageIDs]


        if distributeTraining:
            # retrieve available workers
            i = current_app.control.inspect()
            num_workers = len(i.stats())

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
        

        # initiate job status
        statusFrame = process.freeze()
        self.messages[statusFrame.id] = {
            'type': 'training',
            'submitted': str(current_time()),
            'status': celery.states.PENDING,
            'meta': {'message':'sending job to worker'}
        }

        # submit job
        job = process.apply_async(ignore_result=False, result_extended=True)

        # start listener thread
        t = threading.Thread(target=self._training_initiated, args=(job, distributeTraining,))
        t.start()

        return 'ok' #TODO


    def _do_inference(self, imageIDs, maxNumWorkers=-1):

        # setup
        if maxNumWorkers != 1:
            # only query the number of available workers if more than one is specified to save time
            i = current_app.control.inspect()
            num_workers = len(i.stats())
            if maxNumWorkers == -1:
                maxNumWorkers = num_workers   #TODO: more than one process per worker?
            else:
                maxNumWorkers = min(maxNumWorkers, num_workers)

        # distribute across workers
        images_subset = array_split(imageIDs, max(1, len(imageIDs) // maxNumWorkers))
        jobs = []
        for subset in images_subset:
            job = celery_interface.call_inference.si(imageIDs=subset)
            jobs.append(job)

        # initiate job status
        jobGroup = group(jobs)
        statusFrame = jobGroup.freeze()
        
        self.messages[statusFrame.id] = {
            'type': 'inference',
            'submitted': str(current_time()),
            'status': celery.states.PENDING,
            'meta': {'message':'sending job to worker'}
        }

        # send job
        result = jobGroup.apply_async()

        # start listener thread
        t = threading.Thread(target=self._inference_initiated, args=(result,))
        t.start()
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

        self._do_inference(imageIDs, maxNumWorkers)
        return 'ok' #TODO


    
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

        self._do_inference(imageIDs, maxNumWorkers)
        return 'ok' #TODO
    


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
                self.watchdog.nudge()

                status['project'] = {
                    'num_annotated': self.watchdog.lastCount,
                    'num_next_training': self.watchdog.annoThreshold
                }


        # running tasks status
        if tasks:
            status['tasks'] = {}
            for key in self.messages.keys():
                msg = self.messages[key]

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
            i = current_app.control.inspect()
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