'''
    Performs immediate image querying operations, such as area selection
    (GrabCut, etc.).

    2021 Benjamin Kellenberger
'''

from bottle import request, abort
from .backend.middleware import ImageQueryingMiddleware


class ImageQuerier:

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app

        self.login_check = None

        self.middleware = ImageQueryingMiddleware(config, dbConnector)
        self._initBottle()
    

    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun
    

    def _initBottle(self):

        @self.app.post('/<project>/grabCut')
        def grab_cut(project):
            if not self.loginCheck(extend_session=True):
                abort(401, 'forbidden')
            
            try:
                args = request.json
                imgPath = args['image_path']        #TODO: allow submitting an image directly; query by image ID; etc.
                coords = args['coordinates']
                returnPolygon = args.get('return_polygon', False)
                numIter = args.get('num_iter', 5)

                result = self.middleware.grabCut(project, imgPath, coords, returnPolygon, numIter)
                return {
                    'result': result
                }
            
            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }
            
