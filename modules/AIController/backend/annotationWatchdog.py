'''
    Threadable class that periodically polls the database for new annotations
    (i.e., images that have been screened since the creation date of the last
    model state).
    If the number of newly screened images reaches or exceeds a threshold as
    defined in the configuration file, a 'train' task is submitted to the Cele-
    ry worker(s) and this thread is terminated.

    2019-21 Benjamin Kellenberger
'''

import json
from threading import Thread, Event
import math
from psycopg2 import sql
from celery import current_app

from ..taskWorkflow import task_ids_match



class Watchdog(Thread):

    def __init__(self, project, config, dbConnector, middleware):
        super(Watchdog, self).__init__()
        self.project = project
        self.config = config
        self.dbConnector = dbConnector
        self.middleware = middleware

        self.timer = Event()
        self._stop_event = Event()

        # waiting times
        self.maxWaitingTime = 1800                      # seconds
        self.minWaitingTime = 20
        self.currentWaitingTime = self.minWaitingTime   # modulated based on progress and activity

        self.lastCount = 0                              # for difference tracking

        self._load_properties()
        self._check_ongoing_tasks()

    

    def _check_ongoing_tasks(self):
        #TODO: limit to AIModel tasks
        self.runningTasks = []
        tasksRunning_db = {}
        queryResult = None
        try:
            queryResult = self.dbConnector.execute(
                sql.SQL('''
                    SELECT id, tasks, timeFinished, succeeded, abortedBy
                    FROM {}
                    WHERE launchedBy IS NULL AND timeFinished IS NULL;
                ''').format(sql.Identifier(self.project, 'workflowhistory')),
                None, 'all'
            )
        except Exception as e:
            # couldn't query database anymore, assume project is dead and kill watchdog
            self.stop()
            return
        if queryResult is not None:
            for task in queryResult:
                #TODO: fields to choose?
                if task['timefinished'] is None and \
                    task['abortedby'] is None:
                    tasksRunning_db[task['id']] = json.loads(task['tasks'])
        
        tasks_orphaned = set()
        activeTasks = current_app.control.inspect().active()
        if not isinstance(activeTasks, dict):
            activeTasks = {}
        for dbKey in tasksRunning_db.keys():
            # auto-launched workflow running according to database; check Celery for completeness
            if not len(activeTasks):
                # no task is running; flag all in DB as "orphaned"
                tasks_orphaned.add(dbKey)
            
            else:
                for key in activeTasks:
                    for task in activeTasks[key]:
                        # check type of task
                        taskName = task['name'].lower()
                        if not (taskName.startswith('aiworker') or \
                            taskName in ('aicontroller.get_training_images', 'aicontroller.get_inference_images')):
                            # task is not AI model training-related; skip
                            continue

                        if task_ids_match(tasksRunning_db[dbKey], task['id']):
                            # confirmed task running
                            self.runningTasks.append(task['id'])
                        else:
                            # task not running; check project and flag as such in database
                            try:
                                project = task['kwargs']['project']
                                if project == self.project:
                                    tasks_orphaned.add(dbKey)
                            except:
                                continue
        
        # vice-versa: check running tasks and re-enable them if flagged as orphaned in DB
        tasks_resurrected = set()
        for key in activeTasks:
            for task in activeTasks[key]:

                # check type of task
                taskName = task['name'].lower()
                if not (taskName.startswith('aiworker') or \
                    taskName in ('aicontroller.get_training_images', 'aicontroller.get_inference_images')):
                    # task is not AI model training-related; skip
                    continue

                try:
                    project = task['kwargs']['project']
                    taskID = task['id']
                    if project == self.project and taskID not in tasksRunning_db:
                        tasks_resurrected.add(taskID)
                        self.runningTasks.append(taskID)
                except:
                    continue
        
        tasks_orphaned = tasks_orphaned.difference(tasks_resurrected)
                
        # clean up orphaned tasks
        if len(tasks_orphaned):
            self.dbConnector.execute(sql.SQL('''
                UPDATE {}
                SET timeFinished = NOW(), succeeded = FALSE,
                    messages = 'Auto-launched task did not finish'
                WHERE id IN %s;
            ''').format(sql.Identifier(self.project, 'workflowhistory')),
                (tuple((t,) for t in tasks_orphaned),), None)

        # resurrect running tasks if needed
        if len(tasks_resurrected):
            self.dbConnector.execute(sql.SQL('''
                UPDATE {}
                SET timeFinished = NULL, succeeded = NULL, messages = NULL
                WHERE id IN %s;
            ''').format(sql.Identifier(self.project, 'workflowhistory')),
                (tuple((t,) for t in tasks_resurrected),), None)


    
    def _load_properties(self):
        '''
            Loads project auto-train properties, such as the number of images
            until re-training, from the database.
        '''
        self.properties = self.dbConnector.execute('SELECT * FROM aide_admin.project WHERE shortname = %s',
                        (self.project,), 1)
        self.properties = self.properties[0]
        if self.properties['numimages_autotrain'] is None:
            # auto-training disabled
            self.properties['numimages_autotrain'] = -1
        if self.properties['minnumannoperimage'] is None:
            self.properties['minnumannoperimage'] = 0
        minNumAnno = self.properties['minnumannoperimage']
        if minNumAnno > 0:
            minNumAnnoString = sql.SQL('''
                WHERE image IN (
                    SELECT cntQ.image FROM (
                        SELECT image, count(*) AS cnt FROM {id_anno}
                        GROUP BY image
                    ) AS cntQ WHERE cntQ.cnt > %s
                )
            ''').format(
                id_anno=sql.Identifier(self.project, 'annotation')
            )
            self.queryVals = (minNumAnno,)
        else:
            minNumAnnoString = sql.SQL('')
            self.queryVals = None
        self.queryStr = sql.SQL('''
            SELECT COUNT(image) AS count FROM (
                SELECT image, MAX(last_checked) AS lastChecked FROM {id_iu}
                {minNumAnnoString}
                GROUP BY image
            ) AS query
            WHERE query.lastChecked > (
                SELECT MAX(timeCreated) FROM (
                    SELECT to_timestamp(0) AS timeCreated
                    UNION (
                        SELECT MAX(timeCreated) AS timeCreated FROM {id_cnnstate}
                    )
            ) AS tsQ);
        ''').format(
            id_iu=sql.Identifier(self.project, 'image_user'),
            id_cnnstate=sql.Identifier(self.project, 'cnnstate'),
            minNumAnnoString=minNumAnnoString)



    def nudge(self):
        '''
            Notifies the watchdog that users are active; in this case the querying
            interval is shortened.
        '''
        self.currentWaitingTime = self.minWaitingTime

    

    def recheckAutotrainSettings(self):
        '''
            Force re-check of auto-train mode of project. To be called whenever
            auto-training is changed through the configuration page.
        '''
        self._load_properties()
        self.nudge()



    def getThreshold(self):
        return self.properties['numimages_autotrain']


    def getAImodelAutoTrainingEnabled(self):
        return self.properties['ai_model_enabled']

    
    
    def getOngoingTasks(self):
        self._check_ongoing_tasks()
        return self.runningTasks


    def stop(self):
        self._stop_event.set()
    

    def stopped(self):
        return self._stop_event.is_set()



    def run(self):

        while True:
            if self.stopped():
                return

            # check if project still exists (TODO: less expensive alternative?)
            projectExists = self.dbConnector.execute('''
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = %s
                        AND table_name = 'workflowhistory'
                    );
                ''', (self.project,), 1)
            if not projectExists[0]['exists']:
                # project doesn't exist anymore; terminate process
                self.stop()
                return

            taskOngoing = (len(self.getOngoingTasks()) > 0)

            if self.getAImodelAutoTrainingEnabled() and self.getThreshold() > 0:

                # check if AIController worker and AIWorker are available
                aiModelInfo = self.middleware.get_ai_model_training_info(self.project)
                hasAICworker = (len(aiModelInfo['workers']['AIController']) > 0)
                hasAIWworker = (len(aiModelInfo['workers']['AIWorker']) > 0)

                # poll for user progress
                count = self.dbConnector.execute(self.queryStr, self.queryVals, 1)
                if count is None:
                    # project got deleted
                    return
                count = count[0]['count']

                if not taskOngoing and \
                    count >= self.properties['numimages_autotrain'] and \
                        hasAICworker and hasAIWworker:
                    # threshold exceeded; load workflow
                    defaultWorkflowID = self.dbConnector.execute('''
                        SELECT default_workflow FROM aide_admin.project
                        WHERE shortname = %s;
                    ''', (self.project,), 1)
                    defaultWorkflowID = defaultWorkflowID[0]['default_workflow']

                    try:
                        if defaultWorkflowID is not None:
                            # default workflow set
                            self.middleware.launch_task(self.project, defaultWorkflowID, None)

                        else:
                            # no workflow set; launch standard training-inference chain
                            self.middleware.start_train_and_inference(project=self.project,
                                minTimestamp='lastState',
                                maxNumImages_train=self.properties['maxnumimages_train'],
                                maxNumWorkers_train=self.config.getProperty('AIController', 'maxNumWorkers_train', type=int, fallback=1),           #TODO: replace by project-specific argument
                                forceUnlabeled_inference=True,
                                maxNumImages_inference=self.properties['maxnumimages_inference'],
                                maxNumWorkers_inference=self.config.getProperty('AIController', 'maxNumWorkers_inference', type=int, fallback=-1))  #TODO: replace by project-specific argument
                            
                    except:
                        # error in case auto-launched task is already ongoing; ignore
                        pass
            
                else:
                    # users are still labeling; update waiting time
                    progressPerc = count / self.properties['numimages_autotrain']
                    waitTimeFrac = (0.8*(1 - math.pow(progressPerc, 4))) + \
                                                (0.2 * (1 - math.pow((count - self.lastCount)/max(1, count + self.lastCount), 2)))

                    self.currentWaitingTime = max(self.minWaitingTime, min(self.maxWaitingTime, self.maxWaitingTime * waitTimeFrac))
                    self.lastCount = count


            # wait in intervals to be able to listen to nudges
            secondsWaited = 0
            while secondsWaited < self.currentWaitingTime:
                self.timer.wait(secondsWaited)
                secondsWaited += 10     # be able to respond every ten seconds