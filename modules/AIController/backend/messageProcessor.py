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

    2019-21 Benjamin Kellenberger
'''

from threading import Thread, Event
import time
from datetime import datetime
import celery
from util.helpers import current_time


#TODO: class is in principle deprecated; only functionality that needs to be moved is the polling of the workers
class MessageProcessor(Thread):

    def __init__(self, celery_app):
        super(MessageProcessor, self).__init__()
        
        self.celery_app = celery_app
        self._stop_event = Event()

        # job store
        self.jobs = {}          # dict of lists (one list for each project)

        # message store
        self.messages = {}      # dict of dicts (one dict for each project)

        # worker status store
        self.worker_status = {}
    

    def stop(self):
        self._stop_event.set()
    

    def stopped(self):
        return self._stop_event.is_set()


    @staticmethod
    def unpack_chain(nodes): 
        while nodes.parent:
            yield nodes.parent
            nodes = nodes.parent
        yield nodes



    def poll_worker_status(self, pollNow=False):
        if pollNow or not len(self.worker_status):
            self.pollNow()
        return self.worker_status



    # def __poll_tasks(self, project):
    #     status = {}
    #     task_ongoing = False
        
    #     if not project in self.messages:
    #         return status, task_ongoing

    #     for key in self.messages[project].keys():
    #         job = self.messages[project][key]
    #         msg = self.celery_app.backend.get_task_meta(key)
    #         if not len(msg):
    #             continue

    #         # check for worker failures
    #         if msg['status'] == celery.states.FAILURE:
    #             # append failure message
    #             if 'meta' in msg:       #TODO: and isinstance(msg['meta'], BaseException):
    #                 info = { 'message': html.escape(str(msg['meta']))}
    #             else:
    #                 info = { 'message': 'an unknown error occurred'}
    #         else:
    #             info = msg['result']
            
    #         status[key] = {
    #             'type': job['type'],
    #             'submitted': job['submitted'],      #TODO: not broadcast across AIController threads...
    #             'status': msg['status'],
    #             'meta': info
    #         }
    #         if 'subjobs' in job:
    #             subjobEntries = []
    #             for subjob in job['subjobs']:
    #                 if isinstance(subjob, GroupResult):
    #                     entry = {
    #                         'id': subjob.id
    #                     }
    #                     subEntries = []
    #                     for res in subjob.results:
    #                         subEntry = {
    #                             'id': res.id,
    #                             'status': res.status,
    #                             'meta': ('complete' if res.status == 'SUCCESS' else str(res.result))       #TODO
    #                         }
    #                         subEntries.append(subEntry)
    #                     entry['subjobs'] = subEntries
    #                 else:
    #                     entry = {
    #                         'id': subjob.id,
    #                         'status': subjob.status,
    #                         'meta': ('complete' if subjob.status == 'SUCCESS' else str(subjob.result))       #TODO
    #                     }
    #                 subjobEntries.append(entry)
    #             status[key]['subjobs'] = subjobEntries
            
    #         # check if ongoing
    #         result = AsyncResult(key)
    #         if result.ready():                  #TODO: chains somehow get stuck in 'PENDING'...
    #             # done; remove from queue
    #             result.forget()
    #             status[key]['status'] = 'SUCCESS'
    #         elif result.failed():
    #             # failed
    #             result.forget()
    #             status[key]['status'] = 'FAILURE'
    #         else:
    #             task_ongoing = True
    #     return status, task_ongoing



    # def poll_status(self, project):
    #     status, task_ongoing = self.__poll_tasks(project)

    #     # make sure to locally poll for jobs not in current AIController thread's stack
    #     #TODO: could be a bit too expensive...
    #     if not task_ongoing:
    #         self.poll_worker_status(project)
    #         status, _ = self.__poll_tasks(project)
    #     return status


    # def register_job(self, project, job, taskType):

    #     if not project in self.jobs:
    #         self.jobs[project] = []

    #     self.jobs[project].append(job)

    #     if not project in self.messages:
    #         self.messages[project] = {}

    #     #TODO: incomplete & probably buggy
    #     # add job with its children
    #     message = {
    #         'id': job.id,
    #         'type': taskType,
    #         'submitted': str(current_time()),
    #         'meta': {'message':'sending job to worker'},
    #         'subjobs': {}
    #     }
    #     if hasattr(job, 'status'):
    #         message['status'] = job.status
    #     if hasattr(job, 'parent'):
    #         subjobs = list(self.unpack_chain(job))
    #         subjobs.reverse()
    #         message['subjobs'] = subjobs
    #     self.messages[project][job.id] = message


    
    # def task_id(self, project):
    #     '''
    #         Returns a UUID that is not already in use.
    #     '''
    #     while True:
    #         id = project + '__' + str(uuid.uuid1())
    #         if project not in self.jobs or id not in self.jobs[project]:
    #             return id


    # def task_ongoing(self, project, taskTypes):
    #     '''
    #         Polls the workers for tasks and returns True if at least
    #         one of the tasks of given type (train, inference, etc.) is
    #         running.
    #     '''

    #     # poll for status
    #     self.pollNow()

    #     # identify types
    #     if isinstance(taskTypes, str):
    #         taskTypes = (taskTypes,)

    #     if not project in self.messages:
    #         return False
    #     for key in self.messages[project].keys():
    #         if self.messages[project][key]['type'] in taskTypes and \
    #             self.messages[project][key]['status'] not in (celery.states.SUCCESS, celery.states.FAILURE,):
    #             print('training ongoing')
    #             return True
    #     return False


    def pollNow(self):
        self.worker_status = {}

        i = self.celery_app.control.inspect()
        stats = i.stats()
        if stats is not None and len(stats):
            active_tasks = i.active()
            if active_tasks is None:
                return

            # worker status
            for key in stats:
                workerName = key.replace('celery@', '')
                activeTasks = []
                if key in active_tasks:
                    for task in active_tasks[key]:
                        # append task if of correct project
                        if 'project' in task['kwargs']:
                            activeTasks.append({
                                'id': task['id'],
                                'project': task['kwargs']['project']
                            })
                            # # also add active tasks to current set if not already there
                            # self.__add_worker_task(task)
                self.worker_status[workerName] = {
                    'active_tasks': activeTasks,
                    # 'scheduled_tasks': scheduled_tasks[key]
                }

            # active tasks
            for key in active_tasks.keys():
                taskList = active_tasks[key]
                for t in taskList:
                    taskID = t['id']
                    taskType = t['name']
                    if 'project' not in t['kwargs']:
                        # non-project-specific task; ignore (TODO)
                        continue
                    project = t['kwargs']['project']

                    if not project in self.messages:
                        self.messages[project] = {}

                    if not taskID in self.messages[project]:
                        # task got lost (e.g. due to server restart); re-add
                        try:
                            timeSubmitted = datetime.fromtimestamp(time.time() - (time.monotonic() - t['time_start']))
                        except Exception:
                            timeSubmitted = current_time() #TODO: dirty hack to make failsafe with UI
                        self.messages[project][taskID] = {
                            'type': taskType,
                            'submitted': str(timeSubmitted),
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
            if self.stopped():
                return

            if not len(self.jobs):
                # no jobs in current queue; ping workers for other running tasks
                self.pollNow()
            
            else:
                time.sleep(10)