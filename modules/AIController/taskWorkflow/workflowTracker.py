'''
    The WorkflowTracker is responsible for launching
    tasks and retrieving (sub-) task IDs for each submitted
    workflow, so that each (sub-) task's status can be queried.
    Launched workflows are also added to the workflow history
    table in the RDB.

    2020-21 Benjamin Kellenberger
'''

from collections.abc import Iterable
from uuid import UUID
import json
from psycopg2 import sql
from celery import current_app
from celery.result import AsyncResult, GroupResult
from celery.task.control import revoke
# from . import task_ids_match



class WorkflowTracker:

    def __init__(self, dbConnector, celeryApp):
        self.dbConnector = dbConnector
        self.celeryApp = celeryApp

        self.activeTasks = {}       # for caching



    @staticmethod
    def getTaskIDs(result):
        def _assemble_ids(result):
            def _get_task_id(result):
                r = {'id': result.id}
                if isinstance(result, GroupResult):
                    r['children'] = []
                    # r['children'] = {}
                    for child in result.children:
                        task, taskID = _get_task_id(child)
                        r['children'].append(task)
                        # r['children'][taskID] = task
                return r, result.id
            tasks = []
            r, _ = _get_task_id(result)
            tasks.append(r)
            if result.parent is not None:
                tasks.extend(_assemble_ids(result.parent))
            return tasks
        
        tasks = _assemble_ids(result)
        tasks.reverse()
        return tasks



    @staticmethod
    def getTasksInfo(tasks, forgetIfFinished=True):
        if tasks is None:
            return None, False, None
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        errors = []
        for t in range(len(tasks)):
            result = AsyncResult(tasks[t]['id'])
            if result.ready():
                tasks[t]['successful'] = result.successful()
                if tasks[t]['successful']:
                    tasks[t]['info'] = None
                else:
                    try:
                        error = str(result.get())
                        errors.append(error)
                    except Exception as e:
                        error = str(e)
                        errors.append(error)
                    tasks[t]['info'] = {}
                    tasks[t]['info']['message'] = error
                if forgetIfFinished:
                    result.forget()
            elif result.info is not None:
                tasks[t]['info'] = result.info
            if result.status is not None:
                tasks[t]['status'] = result.status
            if 'children' in tasks[t]:
                numDone = 0
                numChildren = len(tasks[t]['children'])
                for key in range(numChildren):
                    cResult = AsyncResult(tasks[t]['children'][key]['id'])
                    if cResult.ready():
                        numDone += 1
                        tasks[t]['children'][key]['successful'] = cResult.successful()
                        if tasks[t]['children'][key]['successful']:
                            tasks[t]['children'][key]['info'] = None
                        else:
                            try:
                                error = str(cResult.get())
                                errors.append(error)
                            except Exception as e:
                                error = str(e)
                                errors.append(error)
                            tasks[t]['children'][key]['info'] = {}
                            tasks[t]['children'][key]['info']['message'] = error
                        if forgetIfFinished:
                            cResult.forget()
                    elif cResult.info is not None:
                        tasks[t]['children'][key]['info'] = cResult.info
                    if cResult.status is not None:
                        tasks[t]['children'][key]['status'] = cResult.status
                tasks[t]['num_done'] = numDone
                if numDone == numChildren:
                    tasks[t]['status'] = 'SUCCESSFUL'

        lastResult = AsyncResult(tasks[-1]['id'])
        hasFinished = lastResult.ready()

        return tasks, hasFinished, errors



    @staticmethod
    def _revoke_task(tasks):
        if isinstance(tasks, dict) and 'id' in tasks:
            revoke(tasks['id'], terminate=True)
        elif isinstance(tasks, Iterable):
            for task in tasks:
                WorkflowTracker._revoke_task(task)



    def _cache_task(self, project, taskID, taskDetails):
        if not project in self.activeTasks:
            self.activeTasks[project] = {}
        if isinstance(taskDetails, str):
            taskDetails = json.loads(taskDetails)
        if not isinstance(taskID, str):
            taskID = str(taskID)
        self.activeTasks[project][taskID] = taskDetails
    


    def _remove_from_cache(self, project, taskID):
        if not project in self.activeTasks:
            return
        if not isinstance(taskID, str):
            taskID = str(taskID)
        if not taskID in self.activeTasks[project]:
            return
        del self.activeTasks[project][taskID]



    def launchWorkflow(self, project, task, workflow, author=None):
        '''
            Receives a Celery task chain and the original, unexpanded
            workflow description (see WorkflowDesigner), and launches
            the task chain.
            Unravels the resulting Celery AsyncResult and retrieves all
            (sub-) task IDs and the like, and adds an entry to the data-
            base with it for other workers to be able to query. Stores
            it locally for caching.

            Inputs:
                - project:      Project shortname
                - task:         Celery object (typically a chain) that
                                contains all tasks to be executed in the
                                workflow
                - workflow:     Task workflow description, as acceptable
                                by the WorkflowDesigner. We store the ori-
                                ginal, non-expanded workflows, so that
                                they can be easily reused and visualized
                                from the history, if required.
                - author:       Name of the workflow initiator. May be
                                None if the workflow was auto-launched.

            If the author is None (i.e., workflow was automatically initi-
            ated), the function first checks whether another auto-launched
            workflow is still running for this project. Since only one such
            workflow is permitted to run at a time per project, it then
            refuses to launch another one. Currently running workflows are
            identified through the database and Celery running status, if
            needed.
        '''

        #TODO: we now assume the AIC middleware has checked whether task is allowed to launch
        # if author is None:
        #     # check if workflow is already running
        #     alWFrunning = {}
        #     runningAutoWFs = self.dbConnector.execute(
        #         sql.SQL('''
        #             SELECT id, tasks, timeFinished, succeeded, abortedBy
        #             FROM {}
        #             WHERE launchedBy IS NULL;
        #         ''').format(sql.Identifier(project, 'workflowhistory')),
        #         None, 'all'
        #     )
        #     if runningAutoWFs is not None:
        #         for wf in runningAutoWFs:
        #             #TODO: fields to choose?
        #             if wf['timefinished'] is None and \
        #                 wf['abortedby'] is None:
        #                 alWFrunning[wf['id']] = json.loads(wf['tasks'])
            
        #     alTaskRunning = False
        #     alTasks_orphaned = set()
        #     activeTasks = current_app.control.inspect().active()
        #     for alKey in alWFrunning.keys():
        #         # auto-launched workflow running according to database; check Celery for completeness
        #         for key in activeTasks:
        #             for task in activeTasks[key]:
        #                 if task_ids_match(alWFrunning[alKey], task['id']):
        #                     # confirmed task running
        #                     alTaskRunning = True
                            
        #                 else:
        #                     # task not running; flag as such in database
        #                     alTasks_orphaned.add(alKey)

        #     # clean up orphaned tasks
        #     if len(alTasks_orphaned):
        #         self.dbConnector.execute(sql.SQL('''
        #             UPDATE {}
        #             SET timeFinished = NOW(), succeeded = FALSE,
        #                 messages = '[\'Auto-launched task did not finish\']'
        #             WHERE id IN %s;
        #         ''').format(sql.Identifier(project, 'workflowhistory')),
        #         tuple([UUID(i) for i in alTasks_orphaned]), None)

        #     # abort if auto-launched task running
        #     if alTaskRunning:
        #         raise Exception('Auto-launched workflow already running.')

        # create entry in database
        queryStr = sql.SQL('''
            INSERT INTO {id_wHistory} (workflow, launchedBy)
            VALUES (%s, %s)
            RETURNING id;
        ''').format(
            id_wHistory=sql.Identifier(project, 'workflowhistory')
        )
        res = self.dbConnector.execute(queryStr, (json.dumps(workflow), author,), 1)
        taskID = res[0]['id']

        
        # submit workflow
        taskResult = task.apply_async(task_id=str(taskID),
                        queue='AIWorker',
                        ignore_result=False,
                        # result_extended=True,
                        # headers={'headers':{'project':project,'submitted': str(current_time())}}
                        )
        
        # unravel subtasks with children and IDs
        taskdef = WorkflowTracker.getTaskIDs(taskResult)
        
        # add task names
        tasknames_flat = []
        def _flatten_names(task):
            if hasattr(task, 'tasks'):
                # chain or chord
                for subtask in task.tasks:
                    _flatten_names(subtask)
                
                tasknames_flat.append(task.name)

                if hasattr(task, 'body'):
                    # chord
                    tasknames_flat.append(task.body.name)
            else:
                tasknames_flat.append(task.name)
        _flatten_names(task)
        
        global tIdx
        tIdx = 0
        def _assign_names(taskdef):
            global tIdx
            if isinstance(taskdef, list):
                for td in taskdef:
                    _assign_names(td)
            else:
                if 'children' in taskdef:
                    for child in taskdef['children']:
                        _assign_names(child)
                taskdef['name'] = tasknames_flat[tIdx]
            tIdx += 1
        _assign_names(taskdef)

        # commit to DB
        queryStr = sql.SQL('''
            UPDATE {id_wHistory}
            SET tasks = %s
            WHERE id = %s;
        ''').format(
            id_wHistory=sql.Identifier(project, 'workflowhistory')
        )
        self.dbConnector.execute(queryStr, (json.dumps(taskdef), taskID,), None)
        taskID = str(taskID)

        # cache locally
        self._cache_task(project, taskID, taskdef)

        return taskID



    def pollTaskStatus(self, project, taskID):
        '''
            Receives a project shortname and task ID and queries
            Celery for status updates, including for subtasks.
            If the task has finished, the "forget()" method is
            called to clear the Celery queue.
        '''

        if project not in self.activeTasks or \
            taskID not in self.activeTasks[project]:
            # project not cached; get from database
            queryStr = sql.SQL('''
                SELECT tasks FROM {id_wHistory}
                WHERE id = %s
                ORDER BY timeCreated DESC;
            ''').format(
                id_wHistory=sql.Identifier(project, 'workflowhistory')
            )
            result = self.dbConnector.execute(queryStr, (taskID,), 1)
            tasks = result[0]['tasks']

            # cache locally
            self._cache_task(project, taskID, tasks)
        
        else:
            tasks = self.activeTasks[project][taskID]

        # poll for updates
        tasks, hasFinished, errors = WorkflowTracker.getTasksInfo(tasks, False)

        # commit missing details to database if finished
        if hasFinished:
            queryStr = sql.SQL('''
                UPDATE {id_wHistory}
                SET timeFinished = NOW(),
                succeeded = %s,
                messages = %s
                WHERE id = %s;
            ''').format(
                id_wHistory=sql.Identifier(project, 'workflowhistory')
            )
            self.dbConnector.execute(queryStr,
                (len(errors)==0, json.dumps(errors), taskID), None)

            # remove from Celery and from local cache
            WorkflowTracker.getTasksInfo(tasks, True)
            self._remove_from_cache(project, taskID)

        return tasks



    def getActiveTaskIDs(self, project):
        '''
            Receives a project shortname and queries the
            database for unfinished and running tasks.
            Also caches them locally for faster access.
        '''
        response = []

        queryStr = sql.SQL('''
            SELECT id, tasks FROM {id_wHistory}
            WHERE timeFinished IS NULL
            AND succeeded IS NULL
            AND abortedBy IS NULL
            ORDER BY timeCreated DESC;
        ''').format(
            id_wHistory=sql.Identifier(project, 'workflowhistory')
        )
        result = self.dbConnector.execute(queryStr, None, 'all')

        for r in result:
            taskID = r['id']
            response.append(taskID)
            self._cache_task(project, taskID, r['tasks'])
        
        return response



    def getTasks(self, project, runningOrFinished='both', min_timeCreated=None, limit=None):
        '''
            Retrieves all tasks that have been submitted at some point.
            Inputs:
                - "runningOrFinished":  Whether to retrieve only running ('running'),
                                        only finished ('finished'), or both ('both')
                                        tasks.
                - "min_timeCreated":    Minimum date and time when tasks had been created.
                - "limit":              Limit the number of tasks retrieved.
            
            Returns a list with dict entries for all tasks found.
        '''
        queryCriteria = ''
        queryArgs = []
        runningOrFinished = runningOrFinished.lower()
        if runningOrFinished == 'running':
            queryCriteria = 'WHERE timeFinished IS NULL'
        elif runningOrFinished == 'finished':
            queryCriteria = 'WHERE timeFinished IS NOT NULL'
        if min_timeCreated is not None:
            queryCriteria += ('WHERE ' if not len(queryCriteria) else ' AND ')
            queryCriteria += 'timeCreated > %s'
            queryArgs.append(min_timeCreated)
        if isinstance(limit, int):
            queryCriteria += ' LIMIT %s'
            queryArgs.append(limit)

        queryStr = sql.SQL('''
            SELECT * FROM {id_wHistory}
            {queryCriteria}
            ORDER BY timeCreated DESC;
        ''').format(
            id_wHistory=sql.Identifier(project, 'workflowhistory'),
            queryCriteria=sql.SQL(queryCriteria)
        )
        result = self.dbConnector.execute(queryStr, (None if not len(queryArgs) else tuple(queryArgs)), 'all')
        response = []
        for r in result:
            response.append({
                'id': str(r['id']),
                'launched_by': r['launchedby'],
                'aborted_by': r['abortedby'],
                'time_created': r['timecreated'].timestamp(),
                'time_finished': (r['timefinished'].timestamp() if r['timefinished'] is not None else None),
                'succeeded': r['succeeded'],
                'messages': r['messages'],
                'tasks': (json.loads(r['tasks']) if isinstance(r['tasks'], str) else r['tasks']),
                'workflow': (json.loads(r['workflow']) if isinstance(r['workflow'], str) else r['workflow'])
            })
        return response



    def pollAllTaskStatuses(self, project):
        '''
            Retrieves all running tasks in a project
            and returns their IDs, together with a status
            update for each.
        '''
        activeTasks = self.getTasks(project, runningOrFinished='both')

        for t in range(len(activeTasks)):
            taskID = activeTasks[t]['id']
            chainStatus = self.pollTaskStatus(project, taskID)
            if chainStatus is not None:
                activeTasks[t]['children'] = chainStatus
        
        return activeTasks



    def revokeTask(self, username, project, taskID):
        '''
            Revokes (cancels) an ongoing task.
        '''
        # check if task with ID exists
        if project not in self.activeTasks or \
            taskID not in self.activeTasks[project]:
            # query database
            queryStr = sql.SQL('''
                SELECT tasks FROM {id_wHistory}
                WHERE id = %s;
            ''').format(
                id_wHistory=sql.Identifier(project, 'workflowhistory')
            )
            result = self.dbConnector.execute(queryStr, (taskID,), 1)
            tasks = result[0]['tasks']
        else:
            tasks = self.activeTasks[project][taskID]

        # revoke everything
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    task = json.loads(task)
            WorkflowTracker._revoke_task(task)

        # commit to DB
        queryStr = sql.SQL('''
            UPDATE {id_wHistory}
            SET timeFinished = NOW(),
            succeeded = FALSE,
            abortedBy = %s
            WHERE id = %s;
        ''').format(
            id_wHistory=sql.Identifier(project, 'workflowhistory')
        )
        self.dbConnector.execute(queryStr, (username, taskID), None)

        #TODO: return value?

    

    def deleteWorkflow(self, project, ids, revokeIfRunning=False):
        '''
            Removes workflow entries from the database. Input "ids" may either
            be a UUID, an Iterable of UUIDs, or "all", in which case all workflows
            are removed from the project.
            If "revokeIfRunning" is True, all workflows with given IDs that
            happen to still be running are aborted. Otherwise, their deletion
            is skipped.
        '''
        workflowIDs = []
        if ids == 'all':
            # get all workflow IDs from DB
            if revokeIfRunning:
                runningStr = sql.SQL('')
            else:
                runningStr = sql.SQL('WHERE timefinished IS NOT NULL')

            workflowIDs = self.dbConnector.execute(
                sql.SQL('''
                    SELECT id FROM {id_workflowhistory}
                    {runningStr};
                ''').format(
                    id_workflowhistory=sql.Identifier(project, 'workflowhistory'),
                    runningStr=runningStr
                ),
                None,
                'all'
            )

        else:
            if isinstance(ids, str):
                ids = [UUID(ids)]
            elif not isinstance(ids, Iterable):
                ids = [ids]
            for w in range(len(ids)):
                nextID = ids[w]
                if not isinstance(nextID, UUID):
                    nextID = UUID(nextID)
                workflowIDs.append({'id':nextID})
            
        if revokeIfRunning:
            # no need to commit revoke to DB since we're going to delete the workflows anyway
            WorkflowTracker._revoke_task(workflowIDs)
        
        # delete from database
        self.dbConnector.execute(sql.SQL('''
            DELETE FROM {id_workflowhistory}
            WHERE id IN %s;
            '''
        ).format(
            id_workflowhistory=sql.Identifier(project, 'workflowhistory'),
        ), tuple((w['id'],) for w in workflowIDs))