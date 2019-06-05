'''
    Main Bottle and routings for the AITrainer instance.

    2019 Benjamin Kellenberger
'''

from bottle import post, request, response


class AITrainer:

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self._initBottle()


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




''' Convenience launcher (FOR DEBUGGING ONLY) '''
if __name__ == '__main__':

    import argparse
    from runserver import Launcher

    parser = argparse.ArgumentParser(description='Run CV4Wildlife AL Service.')
    parser.add_argument('--instance', type=str, default='AITrainer', const=1, nargs='?')
    args = parser.parse_args()
    Launcher(args)