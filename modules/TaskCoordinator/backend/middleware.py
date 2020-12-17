'''
    2020 Benjamin Kellenberger
'''

import uuid
import html
import celery
from celery import current_app
from celery.result import AsyncResult


class TaskCoordinatorMiddleware:

    def __init__(self, config):
        self.config = config
        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()

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



    def _poll_broker(self):
        '''
            Function to poll message broker for missing jobs.
            This is a rather expensive operation and is thus only called
            if a job is missing locally.
        '''
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



    def submitJob(self, project, process, queue):
        '''
            Assembles all Celery garnish to dispatch a job
            and registers it for status and result querying.
            Returns the respective job ID.
        '''
        task_id = self._task_id(project)
        job = process.apply_async(task_id=task_id,
                                    queue=queue,
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

        if not project in self.jobs:
            self._poll_broker()
            if not project in self.jobs:
                raise Exception('Project {} not found.'.format(project))
        
        if not jobID in self.jobs[project]:
            self._poll_broker()
            if not jobID in self.jobs[project]:
                raise Exception('Job with ID {} does not exist.'.format(jobID))

        # poll status
        #TODO
        msg = self.celery_app.backend.get_task_meta(jobID)
        if msg['status'] == celery.states.FAILURE:
            # append failure message
            if 'meta' in msg:
                info = { 'message': html.escape(str(msg['meta']))}
            elif 'result' in msg:
                info = { 'message': html.escape(str(msg['result']))}
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