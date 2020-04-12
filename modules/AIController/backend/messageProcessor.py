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

    2019-20 Benjamin Kellenberger
'''

from threading import Thread
import time
import uuid
import html
import celery
from celery.result import AsyncResult, GroupResult
import kombu.five
from util.helpers import current_time


class MessageProcessor(Thread):

    def __init__(self, celery_app):
        super(MessageProcessor, self).__init__()
        
        self.celery_app = celery_app

        # job store
        self.jobs = {}          # dict of lists (one list for each project)

        # message store
        self.messages = {}      # dict of dicts (one dict for each project)


    @staticmethod
    def unpack_chain(nodes): 
        while nodes.parent:
            yield nodes.parent
            nodes = nodes.parent
        yield nodes


    def __add_worker_task(self, task):
        project = task['kwargs']['project']
        if not project in self.messages:
            self.messages[project] = {}

        result = AsyncResult(task['id'])
        if not task['id'] in self.messages[project]:
            try:
                timeSubmitted = datetime.fromtimestamp(time.time() - (kombu.five.monotonic() - t['time_start']))
            except:
                timeSubmitted = str(current_time()) #TODO: dirty hack to make failsafe with UI
            self.messages[project][task['id']] = {
                'type': ('train' if 'train' in task['name'] else 'inference'),        #TODO
                'submitted': timeSubmitted,
                'status': celery.states.PENDING,
                'meta': {'message':'job at worker'}
            }

        #TODO: needed?
        if result.ready():
            result.forget()       


    def poll_worker_status(self, project):
        workerStatus = {}
        i = self.celery_app.control.inspect()
        stats = i.stats()
        if stats is not None and len(stats):
            active_tasks = i.active()
            scheduled_tasks = i.scheduled()
            for key in stats:
                workerName = key.replace('celery@', '')

                activeTasks = []
                if key in active_tasks:
                    for task in active_tasks[key]:

                        # append task if of correct project
                        taskProject = task['kwargs']['project']
                        if taskProject == project:
                            activeTasks.append(task['id'])

                        # also add active tasks to current set if not already there
                        self.__add_worker_task(task)

                workerStatus[workerName] = {
                    'active_tasks': activeTasks,
                    'scheduled_tasks': scheduled_tasks[key]
                }
            
            #TODO
            # # also update local cache for completed tasks
            # if project in self.messages:    #TODO
            #     for key in self.messages[project].keys():
            #         if not key in active_tasks and not key in scheduled_tasks:
            #             # task completed
            #             self.messages[project][key]['status'] = celery.states.SUCCESS    #TODO: what if failed?
        return workerStatus


    def __poll_tasks(self, project):
        status = {}
        task_ongoing = False
        
        if not project in self.messages:
            return status, task_ongoing

        for key in self.messages[project].keys():
            job = self.messages[project][key]
            msg = self.celery_app.backend.get_task_meta(key)
            if not len(msg):
                continue

            # check for worker failures
            if msg['status'] == celery.states.FAILURE:
                # append failure message
                if 'meta' in msg:       #TODO: and isinstance(msg['meta'], BaseException):
                    info = { 'message': html.escape(str(msg['meta']))}
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
            result = AsyncResult(key)
            if result.ready():
                # done; remove from queue
                result.forget()
                status[key]['status'] = 'SUCCESS'
            elif result.failed():
                # failed
                result.forget()
                status[key]['status'] = 'FAILURE'
            else:
                task_ongoing = True
        return status, task_ongoing


    def poll_status(self, project):
        status, task_ongoing = self.__poll_tasks(project)

        # make sure to locally poll for jobs not in current AIController thread's stack
        #TODO: could be a bit too expensive...
        if not task_ongoing:
            self.poll_worker_status(project)
            status, _ = self.__poll_tasks(project)
        return status


    def register_job(self, project, job, taskType):

        if not project in self.jobs:
            self.jobs[project] = []

        self.jobs[project].append(job)

        if not project in self.messages:
            self.messages[project] = {}

        #TODO: incomplete & probably buggy
        # add job with its children
        message = {
            'id': job.id,
            'type': taskType,
            'submitted': str(current_time()),
            'status': job.status,
            'meta': {'message':'sending job to worker'},
            'subjobs': {}
        }
        subjobs = list(self.unpack_chain(job))
        subjobs.reverse()
        for subjob in subjobs:
            if isinstance(subjob, GroupResult):
                entry = {
                    'id': subjob.id
                }
                subEntries = []
                for res in subjob.results:
                    subEntry = {
                        'id': res.id,
                        'status': res.status,
                        'meta': ('complete' if res.status == 'SUCCESS' else res.result)       #TODO
                    }
                    subEntries.append(subEntry)
                entry['subjobs'] = subEntries
            else:
                entry = {
                    'id': subjob.id,
                    'status': subjob.status,
                    'meta': ('complete' if subjob.status == 'SUCCESS' else subjob.result)       #TODO
                }
        message['subjobs'] = subjobs
        self.messages[project][job.id] = message

        # # look out for children (if group result)
        # if hasattr(job, 'children') and job.children is not None:
        #     for child in job.children:
        #         self.messages[project][child.task_id] = {
        #         'type': taskType,
        #         'submitted': str(current_time()),
        #         'status': celery.states.PENDING,
        #         'meta': {'message':'sending job to worker'}
        #     }
        # elif not job.id in self.messages[project]:
        #     # no children; add job itself
        #     self.messages[project][job.id] = {
        #     'type': taskType,
        #     'submitted': str(current_time()),
        #     'status': celery.states.PENDING,
        #     'meta': {'message':'sending job to worker'}
        # }

    
    def task_id(self, project):
        '''
            Returns a UUID that is not already in use.
        '''
        while True:
            id = project + '__' + str(uuid.uuid1())
            if project not in self.jobs or id not in self.jobs[project]:
                return id


    def task_ongoing(self, project, taskTypes):
        '''
            Polls the workers for tasks and returns True if at least
            one of the tasks of given type (train, inference, etc.) is
            running.
        '''

        # poll for status
        self.pollNow()

        # identify types
        if isinstance(taskTypes, str):
            taskTypes = (taskTypes,)

        if not project in self.messages:
            return False
        for key in self.messages[project].keys():
            if self.messages[project][key]['type'] in taskTypes and \
                self.messages[project][key]['status'] not in (celery.states.SUCCESS, celery.states.FAILURE,):
                print('training ongoing')
                return True
        return False


    def pollNow(self):
        i = self.celery_app.control.inspect()
        stats = i.stats()
        if stats is not None and len(stats):
            active_tasks = i.active()
            if active_tasks is None:
                return
            for key in active_tasks.keys():
                taskList = active_tasks[key]
                for t in taskList:
                    taskID = t['id']
                    taskType = t['name']
                    project = t['kwargs']['project']

                    if not project in self.messages:
                        self.messages[project] = {}

                    if not taskID in self.messages[project]:
                        # task got lost (e.g. due to server restart); re-add
                        try:
                            timeSubmitted = datetime.fromtimestamp(time.time() - (kombu.five.monotonic() - t['time_start']))
                        except:
                            timeSubmitted = str(current_time()) #TODO: dirty hack to make failsafe with UI
                        self.messages[project][taskID] = {
                            'type': taskType,
                            'submitted': timeSubmitted,
                            'status': celery.states.PENDING,
                            'meta': {'message':'job at worker'}
                        }
                        job = celery.result.AsyncResult(taskID)  #TODO: task.ready()

                        if not project in self.jobs:
                            self.jobs[project] = []
                        self.jobs[project].append(job)


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