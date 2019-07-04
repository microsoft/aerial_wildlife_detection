'''
    Main Bottle and routings for the AIWorker instance.

    2019 Benjamin Kellenberger
'''

from bottle import post, request, response


class AIWorker:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self._initBottle()


    def _initBottle(self):

        ''' notification listener '''
        @self.app.post('/receive')
        def notification_received():
            #TODO
            if not hasattr(request.query, 'message'):
                response.status = 400

            # forward notification
            message = request.forms.get('message')



            return response