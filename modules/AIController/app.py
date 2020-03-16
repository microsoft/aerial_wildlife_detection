'''
    Main Bottle and routings for the AIController instance.

    2019-20 Benjamin Kellenberger
'''

from bottle import post, request, response, abort
from modules.AIController.backend.middleware import AIMiddleware


class AIController:

    #TODO: relay routings if AIController is on a different machine

    def __init__(self, config, app):
        self.config = config
        self.app = app

        self.middleware = AIMiddleware(config)

        self.login_check = None

        self._init_params()
        self._initBottle()


    def _init_params(self):
        self.minNumAnnoPerImage = self.config.getProperty(self, 'minNumAnnoPerImage', type=int, fallback=0)
        self.maxNumImages_train = self.config.getProperty(self, 'maxNumImages_train', type=int)
        self.maxNumWorkers_train = self.config.getProperty(self, 'maxNumWorkers_train', type=int, fallback=-1)
        self.maxNumWorkers_inference = self.config.getProperty(self, 'maxNumWorkers_inference', type=int, fallback=-1)
        self.maxNumImages_inference = self.config.getProperty(self, 'maxNumImages_inference', type=int)


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        @self.app.get('/<project>/listModelStates')
        def list_model_states(project):
            '''
                Returns a list of saved AI model states'
                metadata for a given project.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            return {'modelStates': self.middleware.listModelStates(project) }

        
        @self.app.post('/<project>/startTraining')
        def start_training(project):
            '''
                Manually requests the AIController to train the model.
                This still only works if there is no training process ongoing.
                Otherwise the request is aborted.
            '''
            if self.loginCheck(project=project, admin=True):
                try:
                    params = request.json
                    if 'minNumAnnoPerImage' in params:
                        minNumAnnoPerImage = int(params['minNumAnnoPerImage'])
                    else:
                        minNumAnnoPerImage = self.minNumAnnoPerImage
                    if 'maxNum_train' in params:
                        maxNumImages_train = int(params['maxNum_train'])
                    else:
                        maxNumImages_train = self.maxNumImages_train

                    status = self.middleware.start_training(project=project,
                                        minTimestamp='lastState', 
                                        minNumAnnoPerImage=minNumAnnoPerImage,
                                        maxNumImages=maxNumImages_train,
                                        maxNumWorkers=self.maxNumWorkers_train)
                except Exception as e:
                    status = str(e)
                return { 'status' : status }

            else:
                abort(401, 'unauthorized')

        
        @self.app.post('/<project>/startInference')
        def start_inference(project):
            '''
                Manually requests the AIController to issue an inference job.
            '''
            if self.loginCheck(project=project, admin=True):
                try:
                    params = request.json
                    if 'maxNum_inference' in params:
                        maxNumImages_inference = int(params['maxNum_inference'])
                    else:
                        maxNumImages_inference = self.maxNumImages_inference    #TODO: project-specific
                    status = self.middleware.start_inference(
                                            project=project,
                                            forceUnlabeled=False,      #TODO 
                                            maxNumImages=maxNumImages_inference,
                                            maxNumWorkers=self.maxNumWorkers_inference)
                except Exception as e:
                    status = str(e)
                return { 'status' : status }
            
            else:
                abort(401, 'unauthorized')


        @self.app.post('/<project>/start')
        def start_model(project):
            '''
                Manually launches one of the model processes (train, inference, both, etc.),
                depending on the provided flags.
            '''
            if self.loginCheck(project=project, admin=True):
                try:
                    params = request.json
                    doTrain = 'train' in params and params['train'] is True
                    doInference = 'inference' in params and params['inference'] is True

                    if 'minNumAnnoPerImage' in params:
                        minNumAnnoPerImage = int(params['minNumAnnoPerImage'])
                    else:
                        minNumAnnoPerImage = self.minNumAnnoPerImage    #TODO
                    if 'maxNum_train' in params:
                        maxNumImages_train = int(params['maxNum_train'])
                    else:
                        maxNumImages_train = self.maxNumImages_train    #TODO
                    if 'maxNum_inference' in params:
                        maxNumImages_inference = int(params['maxNum_inference'])
                    else:
                        maxNumImages_inference = self.maxNumImages_inference    #TODO

                    if doTrain:
                        if doInference:
                            status = self.middleware.start_train_and_inference(
                                    project=project,
                                    minTimestamp='lastState',
                                    minNumAnnoPerImage=minNumAnnoPerImage,
                                    maxNumWorkers_train=self.maxNumWorkers_train,
                                    forceUnlabeled_inference=False,
                                    maxNumImages_inference=maxNumImages_inference,
                                    maxNumWorkers_inference=self.maxNumWorkers_inference)
                        else:
                            status = self.middleware.start_training(
                                    project=project,
                                    minTimestamp='lastState',
                                    minNumAnnoPerImage=minNumAnnoPerImage,
                                    maxNumImages=maxNumImages_train,
                                    maxNumWorkers=self.maxNumWorkers_train)
                    else:
                        status = self.middleware.start_inference(
                                    project=project,
                                    forceUnlabeled=False, 
                                    maxNumImages=maxNumImages_inference, 
                                    maxNumWorkers=self.maxNumWorkers_inference)

                    return { 'status' : status }
                except:
                    abort(400, 'bad request')


        @self.app.get('/<project>/status')
        def check_status(project):
            '''
                Queries the middleware for any ongoing training worker processes
                and returns the status of each in a dict.
            '''
            if self.loginCheck(project=project):
                try:
                    queryProject = 'project' in request.query
                    queryTasks = 'tasks' in request.query
                    queryWorkers = 'workers' in request.query
                    status = self.middleware.check_status(
                        project,
                        queryProject, queryTasks, queryWorkers)
                except Exception as e:
                    status = str(e)
                return { 'status' : status }

            else:
                abort(401, 'unauthorized')


        #TODO: REPLACED WITH GENERIC FN OF ProjectAdministration
        # @self.app.get('/<project>/getAImodelSettings')
        # def get_ai_model_info(project):
        #     '''
        #         Returns the model class and settings for the AI model
        #         and the AL criterion.
        #     '''
        #     if not self.login_check(project=project, admin=True):
        #         abort(401, 'unauthorized')
            
        #     return { 'settings': self.middleware.getProjectModelSettings(project) }

    
        @self.app.get('/getAvailableAImodels')
        def get_available_ai_models():
            '''
                Returns all available AI models (class, name) that are
                installed in this instance of AIDE.
            '''
            if not self.login_check(canCreateProjects=True):
                abort(401, 'unauthorized')
            
            return self.middleware.getAvailableAImodels()