'''
    Class that receives and stores messages from the Celery workers and
    calls callbacks in success and failure cases.
    An instance of this class should always be launched as a singleton
    per job dispatcher, especially if used with the RPC backend.

    This class basically circumvents the issue of transient messages of
    the RPC backend. Ideally, we would like to get status messages from
    every job running at the same time, but we can call the "on_message"
    routine only once per instance, since RPC assigns one queue per
    consumer instead of per task.
    To do so, the MessageProcessor will keep track of incoming messages
    and will assign their IDs to open tasks. One of those tasks will
    be the one to receive the "on_raw_message" callback, thereby updating
    messages from the other running tasks as well. As soon as this task is
    finished, the next available one will be assigned to the routine.
    If another task prematurely finishes, the "forget" method will be called
    on it. (TODO)

    2019 Benjamin Kellenberger
'''

from threading import Thread
import time
import uuid
import cgi
import celery
from celery.result import AsyncResult
import kombu.five
from util.helpers import current_time


class MessageProcessor(Thread):

    def __init__(self, celery_app):
        super(MessageProcessor, self).__init__()
        
        self.celery_app = celery_app

        # job store
        self.jobs = {}          # dict of lists (one list for each project)

        # message store
        self.messages = {}

        # callbacks
        self.on_complete = {}


    def __add_worker_task(self, task):
        result = AsyncResult(task['id'])
        if not task['id'] in self.messages:
            try:
                timeSubmitted = datetime.fromtimestamp(time.time() - (kombu.five.monotonic() - t['time_start']))
            except:
                timeSubmitted = str(current_time()) #TODO: dirty hack to make failsafe with UI
            self.messages[task['id']] = {
                'type': ('train' if 'train' in task['name'] else 'inference'),        #TODO
                'submitted': timeSubmitted,
                'status': celery.states.PENDING,
                'meta': {'message':'job at worker'}
            }

        #TODO: needed?
        if result.ready():
            result.forget()       


    def poll_worker_status(self, project):
        #TODO: project
        
        workerStatus = {}
        i = self.celery_app.control.inspect()
        stats = i.stats()
        if stats is not None and len(stats):
            active_tasks = i.active()
            scheduled_tasks = i.scheduled()
            for key in stats:
                workerName = key.replace('celery@', '')

                activeTasks = []
                for task in active_tasks[key]:
                    activeTasks.append(task['id'])

                    # also add active tasks to current set if not already there
                    self.__add_worker_task(task)

                workerStatus[workerName] = {
                    'active_tasks': activeTasks,
                    'scheduled_tasks': scheduled_tasks[key]
                }
            
            # also update local cache for completed tasks
            for key in self.messages.keys():
                if not key in active_tasks and not key in scheduled_tasks:
                    # task completed
                    self.messages[key]['status'] = celery.states.SUCCESS    #TODO: what if failed?
        return workerStatus


    def __poll_tasks(self):
        status = {}
        task_ongoing = False
        for key in self.messages.keys():
            job = self.messages[key]
            msg = self.celery_app.backend.get_task_meta(key)
            if not len(msg):
                continue

            # check for worker failures
            if msg['status'] == celery.states.FAILURE:
                # append failure message
                if 'meta' in msg:       #TODO: and isinstance(msg['meta'], BaseException):
                    info = { 'message': cgi.escape(str(msg['meta']))}
                else:
                    info = { 'message': 'an unknown error occurred'}
            else:
                info = msg['result']
            
            status[key] = {
                'type': job['type'],
                'submitted': job['submitted'],      #TODO: not broadcast across AIController threads...
                'status': msg['status'],
                'meta': info
            }
            
            # check if ongoing
            if not AsyncResult(key).ready():
                task_ongoing = True
        return status, task_ongoing


    def poll_status(self):
        status, task_ongoing = self.__poll_tasks()

        # make sure to locally poll for jobs not in current AIController thread's stack
        #TODO: could be a bit too expensive...
        if not task_ongoing:
            self.poll_worker_status()
            status, _ = self.__poll_tasks()
        return status


    def register_job(self, job, taskType, on_complete=None):
        self.jobs.append(job)

        # look out for children (if group result)
        if hasattr(job, 'children') and job.children is not None:
            for child in job.children:
                self.messages[child.task_id] = {
                'type': taskType,
                'submitted': str(current_time()),
                'status': celery.states.PENDING,
                'meta': {'message':'sending job to worker'}
            }
        elif not job.id in self.messages:
            # no children; add job itself
            self.messages[job.id] = {
            'type': taskType,
            'submitted': str(current_time()),
            'status': celery.states.PENDING,
            'meta': {'message':'sending job to worker'}
        }

        self.on_complete[job.id] = on_complete

    
    def task_id(self):
        '''
            Returns a UUID that is not already in use.
        '''
        while True:
            id = str(uuid.uuid1())
            if id not in self.jobs:
                return id


    def task_ongoing(self, taskType):
        '''
            Polls the workers for tasks and returns True if at least
            one of the tasks of given type (train, inference, etc.) is
            running.
        '''
        # poll for status
        self.pollNow()

        # identify types
        for key in self.messages.keys():
            if self.messages[key]['type'] == taskType and \
                self.messages[key]['status'] not in (celery.states.SUCCESS, celery.states.FAILURE,):
                print('training ongoing')
                return True
        return False


    def pollNow(self):
        i = self.celery_app.control.inspect()
        stats = i.stats()
        if stats is not None and len(stats):
            active_tasks = i.active()
            for key in active_tasks.keys():
                taskList = active_tasks[key]
                for t in taskList:
                    taskID = t['id']
                    if not taskID in self.messages:
                        # task got lost (e.g. due to server restart); re-add
                        try:
                            timeSubmitted = datetime.fromtimestamp(time.time() - (kombu.five.monotonic() - t['time_start']))
                        except:
                            timeSubmitted = str(current_time()) #TODO: dirty hack to make failsafe with UI
                        self.messages[taskID] = {
                            'type': ('train' if 'train' in t['name'] else 'inference'),        #TODO
                            'submitted': timeSubmitted,
                            'status': celery.states.PENDING,
                            'meta': {'message':'job at worker'}
                        }
                        job = celery.result.AsyncResult(taskID)  #TODO: task.ready()
                        self.jobs.append(job)


    def run(self):

        # iterate over all registered tasks and get their result, one by one
        while True:
            if not len(self.jobs):
                # no jobs in current queue; ping workers for other running tasks
                self.pollNow()

            if not len(self.jobs):
                # still no jobs in queue; wait and then try again
                while True:
                    # check if anything in list
                    if len(self.jobs):
                        break
                    else:
                        time.sleep(10)
            
            else:
                nextJob = self.jobs.pop()
                nextJob.get(propagate=True)

                # job finished; handle success and failure cases
                if nextJob.id in self.on_complete:
                    callback = self.on_complete[nextJob.id]
                    if callback is not None:
                        callback(nextJob)

                nextJob.forget()