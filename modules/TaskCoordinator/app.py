'''
    Module responsible for Celery task status polling
    of various other modules that make use of the message
    queue system.

    2020-21 Benjamin Kellenberger
'''

from bottle import request, abort
from .backend.middleware import TaskCoordinatorMiddleware


class TaskCoordinator:

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app
        self.middleware = TaskCoordinatorMiddleware(self.config, dbConnector)
        
        self.login_check = None
        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        ''' Status polling '''
        @self.app.post('/<project>/pollStatus')
        def poll_status_project(project):
            '''
                Receives a task ID and polls the middleware
                for an ongoing data administration task.
                Returns a dict with (meta-) data, including
                the Celery status type, result (if completed),
                error message (if failed), etc.
            '''
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                taskID = request.json['taskID']
                status = self.middleware.pollStatus(project, taskID)
                return {'response': status}

            except Exception as e:
                abort(400, str(e))


        #TODO:
        # @self.app.post('/allTaskStatuses')
        # def poll_status_tasks():
        #     '''
        #         Returns the status of all running tasks.
        #         Only accessible to super users.
        #     '''
        #     if not self.loginCheck(superuser=True):
        #         abort(401, 'forbidden')
    
        #     try:
        #         status = self.middleware.pollStatus(None, None)
        #         return {'response': status}
        #     except Exception as e:
        #         abort(400, str(e))



    def submitJob(self, project, username, process, queue):
        return self.middleware.submitJob(project, username, process, queue)