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
import celery
import kombu.five
from util.helpers import current_time


class MessageProcessor(Thread):

    def __init__(self, celery_app):
        super(MessageProcessor, self).__init__()
        
        self.celery_app = celery_app

        # job store
        self.jobs = []

        # message store
        self.messages = {}

        # callbacks
        self.on_complete = {}


    
    def _on_raw_message(self, message):
        id = message['task_id']
        self.messages[id]['status'] = message['status']
        if 'result' in message:
            self.messages[id]['meta'] = message['result']
        else:
            self.messages[id]['meta'] = None


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
    

    def run(self):

        # iterate over all registered tasks and get their result, one by one
        while True:
            if not len(self.jobs):
                # no jobs in current queue; ping workers for other running tasks
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
                nextJob.get(on_message=self._on_raw_message, propagate=False)

                # job finished; handle success and failure cases
                if nextJob.id in self.on_complete:
                    callback = self.on_complete[nextJob.id]
                    if callback is not None:
                        callback(nextJob)