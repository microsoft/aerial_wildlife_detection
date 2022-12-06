'''
    Main Bottle and routings for the AIController instance.

    2019-21 Benjamin Kellenberger
'''

import html
from bottle import request, abort
from modules.AIController.backend.middleware import AIMiddleware
from modules.AIController.backend import celery_interface
from util.helpers import LogDecorator, parse_boolean


class AIController:

    #TODO: relay routings if AIController is on a different machine

    def __init__(self, config, app, dbConnector, taskCoordinator, verbose_start=False, passive_mode=False):
        self.config = config
        self.app = app

        if verbose_start:
            print('AIController'.ljust(LogDecorator.get_ljust_offset()), end='')

        try:
            self.middleware = AIMiddleware(config, dbConnector, taskCoordinator, passive_mode)
            self.login_check = None
            self._initBottle()
        except Exception as exc:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not launch AIController (message: "{str(exc)}").')

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

            try:
                latest_only = bool(request.params.get('latest_only'))
            except Exception:
                latest_only = False

            return {'modelStates': self.middleware.listModelStates(project, latest_only)}


        @self.app.post('/<project>/deleteModelStates')
        def delete_model_states(project):
            '''
                Receives a list of model IDs and launches a background
                task to delete them.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                username = html.escape(request.get_cookie('username'))
                modelStateIDs = request.json['model_ids']
                taskID = self.middleware.deleteModelStates(project, username, modelStateIDs)
                return {
                    'status': 0,
                    'task_id': taskID
                }

            except Exception as exc:
                return {
                    'status': 1,
                    'message': str(exc)
                }


        @self.app.post('/<project>/duplicateModelState')
        def duplicate_model_state(project):
            '''
                Receives a model state ID and creates a copy of it in this project.
                This copy receives the current date, which makes it the most recent
                model state.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                username = html.escape(request.get_cookie('username'))
                modelStateID = request.json['model_id']
                taskID = self.middleware.duplicateModelState(project, username, modelStateID, skipIfLatest=True)
                return {
                    'status': 0,
                    'task_id': taskID
                }

            except Exception as exc:
                return {
                    'status': 1,
                    'message': str(exc)
                }



        @self.app.get('/<project>/getModelTrainingStatistics')
        def get_model_training_statistics(project):
            '''
                Launches a background task to assemble training
                stats as (optionally) returned by the model across
                all saved model states.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                username = html.escape(request.get_cookie('username'))
                #TODO: permit filtering for model state IDs (change to POST?)
                task_id = self.middleware.getModelTrainingStatistics(project, username, modelStateIDs=None)
                return {
                    'status': 0,
                    'task_id': task_id
                }

            except Exception as exc:
                return {
                    'status': 1,
                    'message': str(exc)
                }



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

            except Exception as exc:
                return { 'status': 1,
                        'message': str(exc)
                }



        @self.app.post('/<project>/abortWorkflow')
        def abort_workflow(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                username = html.escape(request.get_cookie('username'))
                params = request.json
                task_id = params['taskID']
                self.middleware.revoke_task(project, task_id, username)

                return { 'status': 0 }

            except Exception as exc:
                return { 'status': 1,
                        'message': str(exc)
                }



        @self.app.post('/<project>/abortAllWorkflows')
        def abort_all_workflows(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            try:
                username = html.escape(request.get_cookie('username'))
                self.middleware.revoke_all_tasks(project, username)

                return { 'status': 0 }

            except Exception as exc:
                return { 'status': 1,
                        'message': str(exc)
                }



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
                except Exception as exc:
                    status = str(exc)
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
            except Exception as exc:
                return { 'status': str(exc) }



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
                except Exception:
                    workflowID = None
                try:
                    setDefault = request.json['set_default']
                except Exception:
                    setDefault = False

                status = self.middleware.saveWorkflow(project, username, workflow, workflowID, workflowName, setDefault)
                return { 'response': status }

            except Exception as exc:
                return { 'response': {'status':1, 'message':str(exc)} }



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

            except Exception as exc:
                return {'status':1, 'message':str(exc)}



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

            except Exception as exc:
                return {'status':1, 'message':str(exc)}



        @self.app.post('/<project>/deleteWorkflowHistory')
        def delete_workflow_history(project):
            '''
                Receives a string (ID) or list of strings (IDs) for work-
                flow(s) to be deleted. They can only be deleted by the
                authors or else super users.
            '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')

            try:
                workflowID = request.json['workflow_id']
                revokeRunning = (request.json['revoke_running'] if 'revoke_running' in request.json else False)
                status = self.middleware.deleteWorkflow_history(project, workflowID, revokeRunning)
                return status

            except Exception as exc:
                return {'status':1, 'message':str(exc)}



        @self.app.get('/<project>/getAImodelTrainingInfo')
        def get_ai_model_training_info(project):
            '''
            Returns information required to determine whether AI models can be trained
            for a given project.
            This includes:
                - Whether an AI model library is configured for the project
                - Whether at least consumer for each AIController and AIWorker is
                  connected and available
            Returns a dict of this information accordingly.
        '''
            if not self.loginCheck(project, admin=True):
                abort(401, 'unauthorized')

            try:
                status = self.middleware.get_ai_model_training_info(project)
                return {'response': status}

            except Exception as exc:
                return {'status':1, 'message':str(exc)}



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



        @self.app.get('/getAvailableAImodels')
        def get_available_ai_models():
            '''
                Returns all available AI models (class, name) that are
                installed in this instance of AIDE.
            '''
            if not self.loginCheck(canCreateProjects=True):
                abort(401, 'unauthorized')

            return self.middleware.getAvailableAImodels(None)



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
                except Exception:
                    modelLibrary = None
                status = self.middleware.verifyAImodelOptions(project, modelOptions, modelLibrary)
                return {'status': status}
            except Exception as exc:
                return {'status': 1, 'message': str(exc)}



        @self.app.post('/<project>/saveAImodelSettings')
        def save_model_settings(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')

            try:
                settings = request.json['settings']
                response = self.middleware.updateAImodelSettings(project, settings)
                return {'status': 0, 'message': response}
            except Exception as exc:
                return {'status': 1, 'message': str(exc)}



        @self.app.get('/<project>/getLabelclassAutoadaptInfo')
        def get_labelclass_autoadapt_info(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')

            try:
                try:
                    modelID = request.params.get('model_id')
                except Exception:
                    modelID = None
                response = self.middleware.getLabelclassAutoadaptInfo(project, modelID)
                return {'status': 0, 'message': response}
            except Exception as exc:
                return {'status': 1, 'message': str(exc)}



        @self.app.post('/<project>/saveLabelclassAutoadaptInfo')
        def save_labelclass_autoadapt_info(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'unauthorized')

            try:
                enabled = parse_boolean(request.params.get('enabled'))
                response = self.middleware.setLabelclassAutoadaptEnabled(project, enabled)
                return {'status': 0, 'message': response}
            except Exception as exc:
                return {'status': 1, 'message': str(exc)}
