'''
    Module responsible for Celery task status polling
    of various other modules that make use of the message
    queue system.

    2020 Benjamin Kellenberger
'''

from bottle import request, abort
from .backend.middleware import TaskCoordinatorMiddleware


class TaskCoordinator:

    def __init__(self, config, app, verbose_start=False):
        self.config = config
        self.app = app
        self.middleware = TaskCoordinatorMiddleware(self.config)
        
        self.login_check = None
        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        ''' Status polling '''
        @self.app.post('/<project>/pollStatus')
        def pollStatus(project):
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
    

    def submitJob(self, project, process, queue):
        return self.middleware.submitJob(project, process, queue)