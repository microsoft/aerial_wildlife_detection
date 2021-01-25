'''
    Main Bottle and routings for the AIController instance.

    2019-21 Benjamin Kellenberger
'''

import html
from bottle import post, request, response, abort
from modules.AIController.backend.middleware import AIMiddleware
from modules.AIController.backend import celery_interface
from util.helpers import LogDecorator


class AIController:

    #TODO: relay routings if AIController is on a different machine

    def __init__(self, config, app, dbConnector, verbose_start=False, passive_mode=False):
        self.config = config
        self.app = app

        if verbose_start:
            print('AIController'.ljust(LogDecorator.get_ljust_offset()), end='')

        try:
            self.middleware = AIMiddleware(config, dbConnector, passive_mode)
            self.login_check = None
            self._initBottle()
        except Exception as e:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not launch AIController (message: "{str(e)}").')

        if verbose_start:
            LogDecorator.print_status('ok')


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
                Also checks whether model states have been
                shared through the model marketplace.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            return {'modelStates': self.middleware.listModelStates(project) }

        
        #TODO: deprecated; replace with workflow:
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
                        minNumAnnoPerImage = 0      #TODO
                    if 'maxNum_train' in params:
                        maxNumImages_train = int(params['maxNum_train'])
                    else:
                        maxNumImages_train = -1     #TODO

                    status = self.middleware.start_training(project=project,
                                        minTimestamp='lastState', 
                                        minNumAnnoPerImage=minNumAnnoPerImage,
                                        maxNumImages=maxNumImages_train,
                                        maxNumWorkers=1)
                except Exception as e:
                    status = str(e)
                return { 'status' : status }

            else:
                abort(401, 'unauthorized')

        
        #TODO: deprecated; replace with workflow:
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
                        maxNumImages_inference = -1                                 #TODO
                    status = self.middleware.start_inference(
                                            project=project,
                                            forceUnlabeled=False,      #TODO 
                                            maxNumImages=maxNumImages_inference,
                                            maxNumWorkers=-1)           #TODO
                except Exception as e:
                    status = str(e)
                return { 'status' : status }
            
            else:
                abort(401, 'unauthorized')


        #TODO: deprecated; replace with workflow:
        @self.app.post('/<project>/start')
        def start_model(project):
            '''
                Manually launches one of the model processes (train, inference, both, etc.),
                depending on the provided flags.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                params = request.json
                doTrain = 'train' in params and params['train'] is True
                doInference = 'inference' in params and params['inference'] is True

                if 'minNumAnnoPerImage' in params:
                    minNumAnnoPerImage = int(params['minNumAnnoPerImage'])
                else:
                    minNumAnnoPerImage = 0    #TODO
                if 'maxNum_train' in params:
                    maxNumImages_train = int(params['maxNum_train'])
                else:
                    maxNumImages_train = -1    #TODO
                if 'maxNum_inference' in params:
                    maxNumImages_inference = int(params['maxNum_inference'])
                else:
                    maxNumImages_inference = -1    #TODO

                if doTrain:
                    if doInference:
                        status = self.middleware.start_train_and_inference(
                                project=project,
                                minTimestamp='lastState',
                                minNumAnnoPerImage=minNumAnnoPerImage,
                                maxNumWorkers_train=1,          #TODO
                                forceUnlabeled_inference=False,
                                maxNumImages_inference=maxNumImages_inference,
                                maxNumWorkers_inference=-1)     #TODO
                    else:
                        #TODO: expand to other tasks and requests
                        if self.middleware.task_ongoing(project, ('AIController.start_training',
                                                                'AIWorker.call_train', 'AIWorker.call_average_model_states')):
                            raise Exception('A training process is already ongoing for project "{}".'.format(project))
                        
                        status = self.middleware.start_training(
                                project=project,
                                numEpochs=1,
                                minTimestamp='lastState',
                                minNumAnnoPerImage=minNumAnnoPerImage,
                                maxNumImages=maxNumImages_train,
                                maxNumWorkers=1)                #TODO
                else:
                    status = self.middleware.start_inference(
                                project=project,
                                forceUnlabeled=False, 
                                maxNumImages=maxNumImages_inference, 
                                maxNumWorkers=-1)               #TODO

                return { 'status' : status }
            except Exception as e:
                abort(400, 'bad request')



        @self.app.post('/<project>/launchWorkflow')
        def launch_workflow(project):
            '''
                New way of submitting jobs. This starts entire workflows, which
                can be a chain of multiple training and inference jobs in a row.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                username = html.escape(request.get_cookie('username'))
                params = request.json
                result = self.middleware.launch_task(project, params['workflow'], username)
                return result

            except Exception as e:
                return { 'status': 1,
                        'message': str(e) }


        
        @self.app.post('/<project>/abortWorkflow')
        def abort_workflow(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                username = html.escape(request.get_cookie('username'))
                params = request.json
                taskID = params['taskID']
                self.middleware.revoke_task(project, taskID, username)

                return { 'status': 0 }

            except Exception as e:
                return { 'status': 1,
                        'message': str(e) }


        
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
                    nudgeWatchdog = 'nudge_watchdog' in request.query
                    recheckAutotrainSettings = 'recheck_autotrain_settings' in request.query
                    status = self.middleware.check_status(
                        project,
                        queryProject, queryTasks, queryWorkers, nudgeWatchdog, recheckAutotrainSettings)
                except Exception as e:
                    status = str(e)
                return { 'status' : status }

            else:
                abort(401, 'unauthorized')



        @self.app.get('/<project>/getSavedWorkflows')
        def get_saved_workflows(project):
            '''
                Returns all the model workflows saved for this project,
                also made by other users.
            '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                workflows = self.middleware.getSavedWorkflows(project)
                return { 'workflows': workflows }
            except Exception as e:
                return { 'status': str(e) }

        
        @self.app.post('/<project>/saveWorkflow')
        def save_workflow(project):
            '''
                Receives a workflow definition through JSON, verifies it
                by parsing, and stores it in the database if valid. If
                the flag "set_default" is given and set to True, the pro-
                vided workflow will be set as the default, to be executed
                automatically.
            '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                username = html.escape(request.get_cookie('username'))
                workflow = request.json['workflow']
                workflowName = request.json['workflow_name']
                try:
                    # for updating existing workflows
                    workflowID = request.json['workflow_id']
                except:
                    workflowID = None
                try:
                    setDefault = request.json['set_default']
                except:
                    setDefault = False
                
                status = self.middleware.saveWorkflow(project, username, workflow, workflowID, workflowName, setDefault)
                return { 'response': status }

            except Exception as e:
                return { 'response': {'status':1, 'message':str(e)} }


        
        @self.app.post('/<project>/setDefaultWorkflow')
        def set_default_workflow(project):
            '''
                Receives a string (ID) of a workflow and sets it as default
                for a given project.
            '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                workflowID = request.json['workflow_id']
                
                status = self.middleware.setDefaultWorkflow(project, workflowID)
                return status

            except Exception as e:
                return {'status':1, 'message':str(e)}




        @self.app.post('/<project>/deleteWorkflow')
        def delete_workflow(project):
            '''
                Receives a string (ID) or list of strings (IDs) for work-
                flow(s) to be deleted. They can only be deleted by the
                authors or else super users.
            '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                username = html.escape(request.get_cookie('username'))
                workflowID = request.json['workflow_id']
                status = self.middleware.deleteWorkflow(project, username, workflowID)
                return status

            except Exception as e:
                return {'status':1, 'message':str(e)}

    
        @self.app.get('/<project>/getAvailableAImodels')
        def get_available_ai_models(project):
            '''
                Returns all available AI models (class, name) that are
                installed in this instance of AIDE and compatible with
                the project's annotation and prediction types.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')
            
            return self.middleware.getAvailableAImodels(project)


        @self.app.post('/<project>/verifyAImodelOptions')
        def verify_model_options(project):
            '''
                Receives JSON-encoded options and verifies their
                correctness with the AI model (either specified through
                the JSON arguments, or taken from the default project
                option). If the AI model does not support verification
                (as is the case in legacy models), a warning is returned.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                modelOptions = request.json['options']
                try:
                    modelLibrary = request.json['ai_model_library']
                except:
                    modelLibrary = None
                status = self.middleware.verifyAImodelOptions(project, modelOptions, modelLibrary)
                return {'status': status}
            except Exception as e:
                return {'status': 1, 'message': str(e)}


        
        @self.app.post('/<project>/saveAImodelSettings')
        def save_model_settings(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')
            
            try:
                settings = request.json['settings']
                response = self.middleware.updateAImodelSettings(project, settings)
                return {'status': 0, 'message': response}
            except Exception as e:
                return {'status': 1, 'message': str(e)}