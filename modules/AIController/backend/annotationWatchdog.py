'''
    Threadable class that periodically polls the database for new annotations
    (i.e., images that have been screened since the creation date of the last
    model state).
    If the number of newly screened images reaches or exceeds a threshold as
    defined in the configuration file, a 'train' task is submitted to the Cele-
    ry worker(s) and this thread is terminated.

    2019 Benjamin Kellenberger
'''

from threading import Thread, Event
import math
from psycopg2 import sql


class Watchdog(Thread):

    def __init__(self, project, config, dbConnector, middleware):
        super(Watchdog, self).__init__()
        self.project = project
        self._stop_event = Event()

        self.config = config
        self.dbConnector = dbConnector
        self.middleware = middleware

        # initialize properties
        self.properties = self.dbConnector.execute('SELECT * FROM aide_admin.project WHERE shortname = %s',
                        (project,), 1)
        self.properties = self.properties[0]
        if self.properties['numimages_autotrain'] is None:
            # auto-training disabled
            self.properties['numimages_autotrain'] = -1
        if self.properties['minnumannoperimage'] is None:
            self.properties['minnumannoperimage'] = 0

        self.maxWaitingTime = 1800                      # seconds
        self.minWaitingTime = 20
        self.currentWaitingTime = self.minWaitingTime   # modulated based on progress and activity

        self.lastCount = 0                              # for difference tracking

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
                id_anno=sql.Identifier(project, 'annotation')
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
            id_iu=sql.Identifier(project, 'image_user'),
            id_cnnstate=sql.Identifier(project, 'cnnstate'),
            minNumAnnoString=minNumAnnoString)

    
    def stop(self):
        self._stop_event.set()


    def stopped(self):
        return self._stop_event.is_set()


    def nudge(self):
        '''
            Notifies the watchdog that users are active; in this case the querying
            interval is shortened.
        '''
        self.currentWaitingTime = self.minWaitingTime   #TODO


    def getThreshold(self):
        return self.properties['numimages_autotrain']


    def run(self):

        while True:

            # check if training process has already been started or auto-training is disabled
            #TODO: replace "training" flag with project-specific one
            if self.middleware.training or self._stop_event.is_set() or self.properties['numimages_autotrain'] == -1:
                break

            # poll database
            count = self.dbConnector.execute(self.queryStr, self.queryVals, 1)
            count = count[0]['count']

            if count >= self.properties['numimages_autotrain']:
                # threshold exceeded; initiate training process followed by inference and return
                #TODO: 1. replace with default workflow; 2. check beforehand whether default workflow is already running
                self.middleware.start_train_and_inference(project=self.project,
                    minTimestamp='lastState',
                    maxNumImages_train=self.properties['maxnumimages_train'],
                    maxNumWorkers_train=self.config.getProperty('AIController', 'maxNumWorkers_train', type=int, fallback=1),           #TODO: replace by project-specific argument
                    forceUnlabeled_inference=True,
                    maxNumImages_inference=self.properties['maxnumimages_inference'],
                    maxNumWorkers_inference=self.config.getProperty('AIController', 'maxNumWorkers_inference', type=int, fallback=-1))  #TODO: replace by project-specific argument
                self.stop()
                break
            
            else:
                # update waiting time
                progressPerc = count / self.properties['numimages_autotrain']
                waitTimeFrac = (0.8*(1 - math.pow(progressPerc, 4))) + \
                                            (0.2 * (1 - math.pow((count - self.lastCount)/max(1, count + self.lastCount), 2)))

                self.currentWaitingTime = max(self.minWaitingTime, min(self.maxWaitingTime, self.maxWaitingTime * waitTimeFrac))
                self.lastCount = count

                # wait in intervals to be able to listen to nudges
                secondsWaited = 0
                while secondsWaited < self.currentWaitingTime:
                    self._stop_event.wait(secondsWaited)
                    secondsWaited += 10     # be able to respond every ten seconds

        return