'''
    2020-21 Benjamin Kellenberger
'''

import uuid
import html
from psycopg2 import sql
import celery
from celery import current_app
from celery.result import AsyncResult

from modules.Database.app import Database


class TaskCoordinatorMiddleware:

    def __init__(self, config):
        self.config = config
        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()

        self.dbConnector = Database(config)

        self.jobs = {}      # dict per project of jobs



    def _register_job(self, project, username, jobID, jobDescription):
        '''
            Adds a job with its respective ID to the database and
            local dict of running jobs.
        '''
        # add to database
        self.dbConnector.execute(sql.SQL('''
                INSERT INTO {} (task_id, launchedBy, processDescription)
                VALUES (%s, %s, %s);
            ''').format(sql.Identifier(project, 'taskhistory')),
            (jobID, username, jobDescription)
        )

        # cache locally
        if project not in self.jobs:
            self.jobs[project] = set()
        self.jobs[project].add(jobID)


    
    def _update_job(self, project, jobID, abortedBy=None, result=None):
        self.dbConnector.execute(sql.SQL('''
            UPDATE {id_taskhistory}
            SET abortedBy = %s, result = %s, timeFinished = NOW()
            WHERE task_id = (
                SELECT task_id FROM {id_taskhistory}
                WHERE task_id = %s
                ORDER BY timeCreated DESC
                LIMIT 1
            );
            ''').format(id_taskhistory=sql.Identifier(project, 'taskhistory')),
            (abortedBy, str(result), jobID)
        )



    def _task_id(self, project):
        '''
            Returns a UUID that is not already in use.
        '''
        while True:
            id = project + '__' + str(uuid.uuid1())
            if project not in self.jobs or id not in self.jobs[project]:
                return id


        
    def _poll_database(self, project):
        jobIDs = self.dbConnector.execute(
            sql.SQL('''
                SELECT task_id FROM {};
            ''').format(sql.Identifier(project, 'taskhistory')),
            None,
            'all'
        )
        if jobIDs is not None:
            jobIDs = set([j['task_id'] for j in jobIDs])
            
            # cache locally
            if not project in self.jobs:
                self.jobs[project] = set()
            self.jobs[project] = self.jobs[project].union(jobIDs)



    # def _poll_broker(self):
    #     '''
    #         Function to poll message broker for missing jobs.
    #         This is a rather expensive operation and is thus only called
    #         if a job is missing locally.
    #     '''
    #     #TODO: function is obsolete since every task is being registered in database before submission
    #     i = self.celery_app.control.inspect()
    #     stats = i.stats()
    #     if stats is not None and len(stats):
    #         active_tasks = i.active()
    #         for key in stats:
    #             for task in active_tasks[key]:
    #                 # append task if of correct project
    #                 taskProject = task['delivery_info']['routing_key']      #TODO
    #                 if taskProject == project:
    #                     if not task['id'] in self.jobs[project]:
    #                         self._register_job(project, None, task['id'])


    
    def pollJobs(self, project=None):
        # self._poll_broker()
        if project is not None:
            self._poll_database(project)



    def submitJob(self, project, username, process, queue):
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
        
        self._register_job(project, username, task_id, str(process))
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

        if project not in self.jobs or jobID not in self.jobs[project]:
            self.pollJobs(project)
        
        if project not in self.jobs:
            raise Exception(f'Project {project} not found.')
        if jobID not in self.jobs[project]:
            raise Exception('Job with ID {} does not exist.'.format(jobID))

        # poll status
        msg = self.celery_app.backend.get_task_meta(jobID)
        if msg['status'] == celery.states.FAILURE:
            # append failure message
            if 'meta' in msg:
                info = { 'message': html.escape(str(msg['meta']))}
            elif 'result' in msg:
                info = { 'message': html.escape(str(msg['result']))}
            else:
                info = { 'message': 'an unknown error occurred'}
            self._update_job(project, jobID, abortedBy=None, result=info)

        else:
            info = msg['result']

            # check if ongoing and remove if done
            result = AsyncResult(jobID)
            if result.ready():
                status['result'] = result.get()
                result.forget()
                self._update_job(project, jobID, abortedBy=None, result=status['result'])

        status['status'] = msg['status']
        status['meta'] = info

        return status