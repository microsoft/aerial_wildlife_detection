'''
    Middleware for AIController: handles requests and updates to and from the database.

    2019 Benjamin Kellenberger
'''

from datetime import datetime
import pytz
import dateutil.parser
import threading        # we can use Threads since we do not need parallel execution for this task
from modules.AIController.backend import celery_interface
from celery import current_app, group
from celery.result import AsyncResult
from modules.AIWorker.backend.worker import functional
from .sql_string_builder import SQLStringBuilder
from modules.Database.app import Database
from util.helpers import array_split


class AIMiddleware():

    def __init__(self, config):
        self.config = config
        self.dbConn = Database(config)
        self.sqlBuilder = SQLStringBuilder(config)

        self.training_workers = None
        self.training_workers_result = None
        self.training = False   # will be set to True once start_training is called (and False as soon as everything about the training process has finished)

        self.inference_workers = group()


    def _training_initiated(self):
        '''
            To be called after a training process has been started.
            Starts a thread that waits for the worker(s) to finish.
            If there is just one worker, the thread waits for it to
            finish.
            If there is more than one worker, the final thread calls
            the 'average_epochs' instruction of the AI model and again
            waits for it to finish.
            After successful training, the 'training' flag will be set
            to False to allow another round of model training.
        '''

        # check workers
        if self.training_workers_result is None:
            #TODO
            print('enabling training again')
            self.training = False
            return
        
        t = threading.Thread(target=self._start_average_epochs)
        t.start()
        return



    def _start_average_epochs(self):
        # collect epochs (and wait for tasks to finish)
        epochs = []
        # for job in self.training_workers:
        #     print('Worker {} finished.'.format(job.id))
        #     epochs.append(job.get())
        epochs = self.training_workers_result.join()

        
        # send job for epoch averaging
        worker = celery_interface.call_average_epochs.delay(epochs)   #TODO

        result = worker.get()
        print('Averaged epochs: {}, num epochs: {}'.format(result, len(epochs)))

        # flush
        self.training_workers_result = None         #TODO: make history?
        self.training_workers = None                #TODO: ditto
        self.training = False

        return result


    
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


        # get image IDs if training is distributed. Otherwise, let the AIWorker do it to minimize message size
        if distributeTraining:
            # query
            sql = self.sqlBuilder.getTimestampQueryString(minTimestamp, order='oldest', limit=None) #TODO

            if isinstance(minTimestamp, datetime):
                imageIDs = self.dbConn.execute(sql, (minTimestamp,), 'all')

            else:
                imageIDs = self.dbConn.execute(sql, None, 'all')

            
            # distribute across workers (TODO: also specify subset size for multiple jobs, even if only one worker?)
            num_workers = 10    #len(current_app.control.inspect().stats().keys())
            images_subset = array_split(imageIDs, max(1, len(imageIDs) // num_workers))
            print('Subset size: {}'.format(len(images_subset)))

            processes = []
            for subset in images_subset:
                print('Next subset length: {}'.format(len(subset)))
                processes.append(celery_interface.call_train.s(subset))
                # process = celery_interface.call_train.delay(subset) #TODO: route to specific worker? http://docs.celeryproject.org/en/latest/userguide/routing.html#manual-routing
                # self.training_workers[process.id] = process

        else:
            # call one worker directly
            # process = celery_interface.call_train.delay(data) #TODO: route to specific worker? http://docs.celeryproject.org/en/latest/userguide/routing.html#manual-routing
            # self.training_workers[process.id] = process
            processes = [celery_interface.call_train.s(imageIDs)]
        
        self.training_workers = group(processes)
        self.training_workers_result = self.training_workers.apply_async()

        # initiate post-submission routine
        self._training_initiated()

        return 'ok' #TODO



    def _do_inference(self, imageIDs, maxNumWorkers=-1):
        # setup
        if maxNumWorkers == -1:
            maxNumWorkers = len(current_app.control.inspect().stats().keys())   #TODO: more than one process per worker?
        else:
            maxNumWorkers = min(maxNumWorkers, len(current_app.control.inspect().stats().keys()))

        # distribute across workers
        images_subset = array_split(imageIDs, max(1, len(imageIDs) // maxNumWorkers))

        for subset in images_subset:
            job = celery_interface.call_inference.delay(subset)
            self.inference_workers[job.id] = job        #TODO: remove job once finished



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
        if maxNumImages == -1:
            maxNumImages = self.config.getProperty('AIController', 'maxNumImages_inference')


        # load the IDs of the images that are being subjected to inference
        sql = self.sqlBuilder.getInferenceQueryString(forceUnlabeled, maxNumImages)
        imageIDs = self.dbConn.execute(sql, None, 'all')


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
    


    def check_status(self, training_workers, inference_workers):
        '''
            Queries the Celery worker results depending on the parameters specified.
            Returns their status accordingly if they exist.
        '''

        statuses = {}

        #TODO: epoch averaging...

        if training_workers and self.training_workers_result is not None:
            for child in self.training_workers_result.children:
                print(vars(child).keys())
                statuses[child.id] = {
                    'type' : 'training',
                    'status' : child.status,
                    'meta': child.info
                }
        
        if inference_workers and len(self.inference_workers):
            for key in self.inference_workers:
                statuses[key] = {
                    'type' : 'inference',
                    'status' : self.inference_workers[key].status,
                    'info': child.info
                }

        return statuses