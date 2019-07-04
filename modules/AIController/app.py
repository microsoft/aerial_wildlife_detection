'''
    Main Bottle and routings for the AIController instance.

    2019 Benjamin Kellenberger
'''

from bottle import post, request, response
from celery import Celery


class AIController:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self._initTaskScheduler()
        self._initBottle()


    def _initTaskScheduler(self):
        self.taskScheduler = Celery('taskTest', backend=self.config.getProperty(self, 'result_backend'), broker=self.config.getProperty(self, 'broker_URL'))

        @self.taskScheduler.task(bind=True)
        def test(self):
            print(self)
            return 'it works'

        
        #TODO
        try:
            from modules.AIController.taskScheduler import add
            result = add.delay(3, 4)
            print(result.backend)
            print(result.get())
        except Exception as err:
            print(err)
            print('---------------')


    def _initBottle(self):

        ''' notification listener '''
        @self.app.post('/notify')
        def notification_received():
            #TODO
            if not hasattr(request.query, 'message'):
                response.status = 400

            # forward notification
            message = request.forms.get('message')

            return response