'''
    Middleware layer for the data administration
    module.
    Responsible for the following tasks and operations:
    - image management: upload and deletion to and from disk
    - annotation and prediction management: up- and download
      of annotations and model predictions

    2020 Benjamin Kellenberger
'''

import uuid
import html
import celery
from celery import current_app
from celery.result import AsyncResult
from . import celery_interface
from .dataWorker import DataWorker
from modules.Database.app import Database


class DataAdministrationMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()

        self.dataWorker = DataWorker(config)

        self.jobs = {}      # dict per project of jobs



    def _register_job(self, project, job, jobID):
        '''
            Adds a job with its respective ID to the dict
            of running jobs.
        '''
        if not project in self.jobs:
            self.jobs[project] = {}
        self.jobs[project][jobID] = job



    def _task_id(self, project):
        '''
            Returns a UUID that is not already in use.
        '''
        while True:
            id = project + '__' + str(uuid.uuid1())
            if project not in self.jobs or id not in self.jobs[project]:
                return id



    def _submit_job(self, project, process):
        '''
            Assembles all Celery garnish to dispatch a job
            and registers it for status and result querying.
            Returns the respective job ID.
        '''
        task_id = self._task_id(project)
        job = process.apply_async(task_id=task_id,
                                    queue=project+'_dataMgmt', #TODO
                                    ignore_result=False,
                                    result_extended=True,
                                    headers={'headers':{}}) #TODO
        
        self._register_job(project, job, task_id)
        return task_id


    
    def pollStatus(self, project, jobID):
        '''
            Queries the dict of registered jobs and polls
            the respective job for status updates, resp.
            final results. Returns the respective data.
            If the job has terminated or failed, it is re-
            moved from the dict.
            If the job cannot be found in the dict, the
            message broker is being queried for potentially
            missing jobs (e.g. due to multi-threaded web
            server processes), and the missing jobs are
            added accordingly. If the job can still not be
            found, an exception is thrown.
        '''
        status = {}

        # to poll message broker for missing jobs
        def _poll_broker():
            i = self.celery_app.control.inspect()
            stats = i.stats()
            if stats is not None and len(stats):
                active_tasks = i.active()
                for key in stats:
                    for task in active_tasks[key]:
                        # append task if of correct project
                        taskProject = task['delivery_info']['routing_key']
                        if taskProject == project:
                            if not task['id'] in self.jobs[project]:
                                self._register_job(project, task, task['id'])       #TODO: not sure if this works...

        if not project in self.jobs:
            _poll_broker()
            if not project in self.jobs:
                raise Exception('Project {} not found.'.format(project))
        
        if not jobID in self.jobs[project]:
            _poll_broker()
            if not jobID in self.jobs[project]:
                raise Exception('Job with ID {} does not exist.'.format(jobID))

        # poll status
        #TODO
        msg = self.celery_app.backend.get_task_meta(jobID)
        if msg['status'] == celery.states.FAILURE:
            # append failure message
            if 'meta' in msg:
                info = { 'message': html.escape(str(msg['meta']))}
            else:
                info = { 'message': 'an unknown error occurred'}
        else:
            info = msg['result']

            # check if ongoing and remove if done
            result = AsyncResult(jobID)
            if result.ready():
                status['result'] = result.get()
                result.forget()

        status['status'] = msg['status']
        status['meta'] = info

        return status



    def listImages(self, project, imageAddedRange=None, lastViewedRange=None,
            viewcountRange=None, numAnnoRange=None, numPredRange=None,
            orderBy=None, order='desc', startFrom=None, limit=None):
        '''
            #TODO: update description
            Returns a list of images, with ID, filename,
            date image was added, viewcount, number of annotations,
            number of predictions, and last time viewed, for a given
            project.
            The list can be filtered by all those properties (e.g. 
            date and time image was added, last checked; number of
            annotations, etc.), as well as limited in length (images
            are sorted by date_added).
        '''
        
        # submit job 
        process = celery_interface.listImages.si(project, imageAddedRange,
                                                lastViewedRange, viewcountRange,
                                                numAnnoRange, numPredRange,
                                                orderBy, order, startFrom, limit)
        
        task_id = self._submit_job(project, process)
        return task_id
    


    def uploadImages(self, project, images, existingFiles='keepExisting'):
        '''
            Image upload is handled directly through the
            dataWorker, without a Celery dispatching bridge.
        '''
        return self.dataWorker.uploadImages(project, images, existingFiles)



    def scanForImages(self, project):
        '''
            #TODO: update description
            Searches the project image folder on disk for
            files that are valid, but have not (yet) been added
            to the database.
            Returns a list of paths with files.
        '''

        # submit job
        process = celery_interface.scanForImages.si(project)

        task_id = self._submit_job(project, process)
        return task_id



    def addExistingImages(self, project, imageList=None):
        '''
            #TODO: update description
            Scans the project folder on the file system
            for images that are physically saved, but not
            (yet) added to the database.
            Adds them to the project's database schema.
            If an imageList iterable is provided, only
            the intersection between identified images on
            disk and in the iterable are added.

            Returns a list of image IDs and file names that
            were eventually added to the project database schema.
        '''

        # submit job
        process = celery_interface.addExistingImages.si(project, imageList)

        task_id = self._submit_job(project, process)
        return task_id


    
    def removeImages(self, project, imageList, forceRemove=False, deleteFromDisk=False):
        '''
            #TODO: update description
            Receives an iterable of image IDs and removes them
            from the project database schema, including associated
            user views, annotations, and predictions made.
            Only removes entries if no user views, annotations, and
            predictions exist, or else if "forceRemove" is True.
            If "deleteFromDisk" is True, the image files are also
            deleted from the project directory on the file system.

            Returns a list of images that were deleted.
        '''

        # submit job
        process = celery_interface.removeImages.si(project,
                                                    imageList,
                                                    forceRemove,
                                                    deleteFromDisk)

        task_id = self._submit_job(project, process)
        return task_id



    def prepareDataDownload(self, project, dataType='annotation', userList=None, dateRange=None):
        '''
            #TODO: update description
            Polls the database for project data according to the
            specified restrictions:
            - dataType: "annotation" or "prediction"
            - userList: for type "annotation": None (all users) or
                        an iterable of user names
            - dateRange: None (all dates) or two values for a mini-
                         mum and maximum timestamp
            
            Creates a file in this machine's temporary directory
            and returns the file name to it.
            Note that in some cases (esp. for semantic segmentation),
            the number of queryable entries may be limited due to
            file size and free disk space restrictions. An upper cei-
            ling is specified in the configuration *.ini file ('TODO')
        '''

        # submit job
        process = celery_interface.prepareDataDownload.si(project,
                                                    dataType,
                                                    userList,
                                                    dateRange)

        task_id = self._submit_job(project, process)
        return task_id