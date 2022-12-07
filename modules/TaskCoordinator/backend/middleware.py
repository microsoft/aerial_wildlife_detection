'''
    2020-21 Benjamin Kellenberger
'''

import uuid
import html
from psycopg2 import sql
import celery
from celery import current_app
from celery.result import AsyncResult

from util.helpers import isAItask


class TaskCoordinatorMiddleware:

    def __init__(self, config, dbConnector):
        self.config = config
        self.celery_app = current_app
        self.celery_app.set_current()
        self.celery_app.set_default()

        self.dbConnector = dbConnector

        self.tasks = {}      # dict per project of tasks



    def _register_task(self, project, username, task_id, task_name, task_description):
        '''
            Adds a task with its respective ID to the database and
            local dict of running tasks.
        '''
        # add to database
        self.dbConnector.execute(sql.SQL('''
                INSERT INTO {} (task_id, launchedBy, taskName, processDescription)
                VALUES (%s, %s, %s, %s);
            ''').format(sql.Identifier(project, 'taskhistory')),
            (uuid.UUID(task_id), username, task_name, task_description)
        )

        # cache locally
        if project not in self.tasks:
            self.tasks[project] = set()
        self.tasks[project].add(task_id)



    def _update_task(self, project, task_id, aborted_by=None, result=None):
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
            (aborted_by, str(result), task_id)
        )



    def _task_id(self, project):
        '''
            Returns a UUID that is not already in use.
        '''
        while True:
            id = str(uuid.uuid1()) #TODO: causes conflict with database format: project + '__' + str(uuid.uuid1())
            if project not in self.tasks or id not in self.tasks[project]:
                return id



    def _poll_database(self, project, task_name=None, running_only=True):
        option_str = ''
        query_args = []
        if task_name is not None:
            option_str = 'WHERE taskName = %s'
            query_args.append(task_name)
        if running_only is True:
            if len(option_str) > 0:
                option_str += ' AND timeFinished IS NULL'
            else:
                option_str = 'WHERE timeFinished IS NULL'
        task_ids = self.dbConnector.execute(
            sql.SQL('''
                SELECT task_id FROM {id_thistory}
                {option_str};
            ''').format(
                id_thistory=sql.Identifier(project, 'taskhistory'),
                option_str=sql.SQL(option_str)
            ),
            tuple(query_args) if len(query_args)>0 else None,
            'all'
        )
        if task_ids is not None:
            task_ids = {j['task_id'] for j in task_ids}

            # cache locally
            if not project in self.tasks:
                self.tasks[project] = set()
            self.tasks[project] = self.tasks[project].union(task_ids)
        return task_ids



    def poll_task_type(self, project, task_name, running_only=True):
        '''
            Queries the database for tasks of a given project and name.
        '''
        return self._poll_database(project, task_name, running_only)


    def submit_task(self, project, username, process, queue):
        '''
            Assembles all Celery garnish to dispatch a task and registers it for
            status and result querying. Returns the respective task ID.
        '''
        task_id = self._task_id(project)
        process.set(queue=queue).apply_async(task_id=task_id,
                                    queue=queue,
                                    ignore_result=False,
                                    result_extended=True,
                                    headers={'headers':{}})
        task_name = process.task
        self._register_task(project, username, task_id, task_name, str(process))
        return task_id



    def revoke_task(self, project, username, task_id, include_ai_tasks=False):
        '''
            Revokes (aborts) one or more ongoing task(s) and sets a flag in the
            database accordingly. "task_id" may either be a single task ID, an
            Iterable of task IDs, or 'all', in which case all task for a given
            project will be revoked. If "include_ai_tasks" is True, any AI
            model-related tasks will also be revoked (if present in the list).
        '''
        if isinstance(task_id, (str, uuid.UUID)):
            task_id = [task_id]

        #TODO: doesn't work that way; also too expensive...
        if task_id[0] == 'all':
            # revoke all tasks; poll broker first
            task_id = []
            i = self.celery_app.control.inspect()
            stats = i.stats()
            if stats is not None and len(stats):
                active_tasks = i.active()
                for key in stats:
                    for task in active_tasks[key]:
                        try:
                            task_project = task['kwargs']['project']
                            if task_project != project:
                                continue
                            task_type = task['name']
                            if not include_ai_tasks and not isAItask(task_type):  #TODO
                                continue
                            task_id.append(task['id'])
                        except Exception:
                            continue

        # filter tasks if needed
        if not include_ai_tasks:
            tasks_revoke = []
            tasks_project = self.tasks[project]
            for t_id in task_id:
                if t_id in tasks_project:
                    #TODO
                    pass
        else:
            tasks_revoke = task_id

        # revoke
        for task_revoke in tasks_revoke:
            if not isinstance(task_revoke, uuid.UUID):
                task_revoke = uuid.UUID(task_revoke)
            try:
                celery.task.control.revoke(task_revoke, terminate=True)
            except Exception as exc:
                print(exc)    #TODO

        # set flag in database
        if len(tasks_revoke) > 0:
            self.dbConnector.execute(sql.SQL('''
                UPDATE {id_taskhistory}
                SET abortedBy = %s
                WHERE task_id IN %s;
            ''').format(id_taskhistory=sql.Identifier(project, 'taskhistory')),
            (username, tuple(tasks_revoke)))



    def revoke_all_tasks(self, project, username, include_ai_tasks=False):
        '''
            Polls all workers for ongoing tasks under a given project and
            revokes them all. Also sets flags in the database accordingly. If
            "include_ai_tasks" is True, any AI model-related tasks will also be
            revoked (if present in the list).
        '''
        return self.revoke_task(project, username, 'all', include_ai_tasks)



    def poll_task_status(self, project, task_id):
        '''
            Queries the dict of registered tasks and polls the respective task
            for status updates, resp. final results. Returns the respective
            data. If the task has terminated or failed, it is re- moved from the
            dict. If the task cannot be found in the dict, the message broker is
            being queried for potentially missing tasks (e.g. due to
            multi-threaded web server processes), and the missing tasks are
            added accordingly. If the task can still not be found, an exception
            is thrown.
        '''
        status = {}

        if project not in self.tasks or task_id not in self.tasks[project]:
            self._poll_database(project)

        #TODO: we temporarily allow querying all tasks without checking...
        # if project not in self.tasks:
        #     raise Exception(f'Project {project} not found.')
        # if task_id not in self.tasks[project]:
        #     raise Exception('Task with ID {} does not exist.'.format(task_id))

        # poll status
        try:
            msg = self.celery_app.backend.get_task_meta(task_id)
            if msg['status'] == celery.states.FAILURE:
                # append failure message
                if 'meta' in msg:
                    info = { 'message': html.escape(str(msg['meta']))}
                elif 'result' in msg:
                    info = { 'message': html.escape(str(msg['result']))}
                else:
                    info = { 'message': 'an unknown error occurred'}
                self._update_task(project, task_id, aborted_by=None, result=info)

            else:
                info = msg['result']

                # check if ongoing and remove if done
                result = AsyncResult(task_id)
                if result.ready():
                    status['result'] = result.get()
                    result.forget()
                    self._update_task(project, task_id, aborted_by=None, result=status['result'])

            status['status'] = msg['status']
            status['meta'] = info

        except Exception:
            status = {}

        return status
