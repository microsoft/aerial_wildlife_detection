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

        self.config = config
        self.dbConnector = dbConnector
        self.middleware = middleware

        # waiting times
        self.maxWaitingTime = 1800                      # seconds
        self.minWaitingTime = 20
        self.currentWaitingTime = self.minWaitingTime   # modulated based on progress and activity

        self.lastCount = 0                              # for difference tracking

        self._load_properties()
        self.taskOngoing = self._check_task_ongoing()

    

    def _check_task_ongoing(self):
        #TODO: WorkflowTracker contains the same code...
        tasksRunning_db = {}
        queryResult = self.dbConnector.execute(
            sql.SQL('''
                SELECT id, tasks, timeFinished, succeeded, abortedBy
                FROM {}
                WHERE launchedBy IS NULL AND timeFinished IS NULL;
            ''').format(sql.Identifier(self.project, 'workflowhistory')),
            None, 'all'
        )
        if queryResult is not None:
            for task in queryResult:
                #TODO: fields to choose?
                if task['timefinished'] is None and \
                    task['abortedby'] is None:
                    tasksRunning_db[task['id']] = json.loads(task['tasks'])
        
        self.taskOngoing = False
        tasks_orphaned = set()
        activeTasks = current_app.control.inspect().active()
        for dbKey in tasksRunning_db.keys():
            # auto-launched workflow running according to database; check Celery for completeness
            if not len(activeTasks):
                # no task is running; flag all in DB as "orphaned"
                tasks_orphaned.add(dbKey)
            
            else:
                for key in activeTasks:
                    for task in activeTasks[key]:
                        if task_ids_match(tasksRunning_db[dbKey], task['id']):
                            # confirmed task running
                            self.taskOngoing = True
                        else:
                            # task not running; flag as such in database
                            tasks_orphaned.add(dbKey)
                
        # clean up orphaned tasks
        if len(tasks_orphaned):
            self.dbConnector.execute(sql.SQL('''
                UPDATE {}
                SET timeFinished = NOW(), succeeded = FALSE,
                    messages = 'Auto-launched task did not finish'
                WHERE id IN %s;
            ''').format(sql.Identifier(self.project, 'workflowhistory')),
                (tuple((t,) for t in tasks_orphaned),), None)

        return self.taskOngoing


    
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



    def getThreshold(self):
        return self.properties['numimages_autotrain']



    def run(self):

        while True:
            
            self.taskOngoing = self._check_task_ongoing()

            if self.properties['numimages_autotrain'] <= 0:
                # auto-training disabled; re-query database, then wait again
                self._load_properties()

            else:
                # poll for user progress
                count = self.dbConnector.execute(self.queryStr, self.queryVals, 1)
                count = count[0]['count']

                if not self.taskOngoing and count >= self.properties['numimages_autotrain']:
                    # threshold exceeded; initiate training process followed by inference and return
                    try:
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