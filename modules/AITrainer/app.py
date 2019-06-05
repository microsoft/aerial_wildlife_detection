'''
    Main Bottle and routings for the AITrainer instance.

    2019 Benjamin Kellenberger
'''

from bottle import Bottle, post, request, response


class AITrainerCommandListener:

    def __init__(self, config):
        self.host = config['AITRAINER']['host']
        self.port = config['AITRAINER']['port']

        self._initBottle()


    def _initBottle(self):

        app = Bottle()

        ''' notification listener '''
        @app.post('/notify')
        def notification_received():
            #TODO
            if not hasattr(request.query, 'message'):
                response.status = 400

            # forward notification
            message = request.forms.get('message')



            return response

