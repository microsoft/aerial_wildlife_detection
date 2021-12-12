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
            # if not self.loginCheck(extend_session=True):
            #     abort(401, 'forbidden')
            
            try:
                args = request.json
                imgPath = args['image_path']
                coords = args['coordinates']
                returnPolygon = args.get('return_polygon', False)
                numIter = args.get('num_iter', 5)

                result = self.middleware.grabCut(project, imgPath, coords, returnPolygon, numIter)
                return {
                    'status': 0,
                    'result': result
                }
            
            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }
        

        @self.app.post('/<project>/magic_wand')
        def magic_wand(project):
            # if not self.loginCheck(extend_session=True):
            #     abort(401, 'forbidden')
            
            try:
                args = request.json
                imgPath = args['image_path']
                seedCoords = args['seed_coordinates']
                tolerance = args.get('tolerance', 32)
                maxRadius = args.get('max_radius', None)
                rgbOnly = args.get('rgb_only', False)

                result = self.middleware.magicWand(project, imgPath, seedCoords, tolerance, maxRadius, rgbOnly)
                return {
                    'status': 0,
                    'result': result
                }

            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }