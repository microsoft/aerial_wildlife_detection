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
# import time
import math


class Watchdog(Thread):

    def __init__(self, config, dbConnector, middleware):
        super(Watchdog, self).__init__()
        self._stop_event = Event()

        self.config = config
        self.dbConnector = dbConnector
        self.middleware = middleware

        # initialize properties
        self.annoThreshold = float(config.getProperty('AIController', 'numImages_autoTrain', type=int))
        self.maxNumImages_train = self.config.getProperty('AIController', 'maxNumImages_train', type=int)
        self.maxNumWorkers_train = self.config.getProperty('AIController', 'maxNumWorkers_train', type=int, fallback=-1)
        self.maxNumWorkers_inference = self.config.getProperty('AIController', 'maxNumWorkers_inference', type=int, fallback=-1)
        self.maxNumImages_inference = self.config.getProperty('AIController', 'maxNumImages_inference', type=int)

        self.maxWaitingTime = 1800                      # seconds
        self.minWaitingTime = 20
        self.currentWaitingTime = self.minWaitingTime   # modulated based on progress and activity

        self.lastCount = 0                              # for difference tracking

        self.sql = '''
            SELECT COUNT(image) AS count FROM (
                SELECT image, MAX(last_checked) AS lastChecked FROM {schema}.image_user
                GROUP BY image
            ) AS query
            WHERE query.lastChecked > (
                SELECT MAX(timeCreated) FROM (
                    SELECT to_timestamp(0) AS timeCreated
                    UNION (
                        SELECT MAX(timeCreated) AS timeCreated FROM {schema}.cnnstate
                    )
            ) AS tsQ);
        '''.format(schema=config.getProperty('Database', 'schema'))

    
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


    def run(self):

        while True:

            # check if training process has already been started
            if self.middleware.training or self._stop_event.is_set():
                break

            # poll database
            count = self.dbConnector.execute(self.sql, None, 1)
            count = count[0]['count']

            if count >= self.annoThreshold:
                # threshold exceeded; initiate training process followed by inference and return
                self.middleware.start_train_and_inference(minTimestamp='lastState',
                    maxNumImages_train=self.maxNumImages_train,
                    maxNumWorkers_train=self.maxNumWorkers_train,
                    forceUnlabeled_inference=True,
                    maxNumImages_inference=self.maxNumImages_inference,
                    maxNumWorkers_inference=self.maxNumWorkers_inference)
                self.stop()
                break
            
            else:
                # update waiting time
                progressPerc = count / self.annoThreshold
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