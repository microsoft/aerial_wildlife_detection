'''
    Bottle routings for labeling statistics of project,
    including per-user analyses and progress.

    2019 Benjamin Kellenberger
'''

from bottle import request, static_file, abort
from .backend.middleware import ProjectStatisticsMiddleware


class ProjectStatistics:

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = 'modules/ProjectStatistics/static'
        self.middleware = ProjectStatisticsMiddleware(config)

        self.login_check = None
        self._initBottle()


    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        @self.app.route('/statistics/<filename:re:.*>') #TODO: /statistics/static/ is ignored by Bottle...
        def send_static(filename):
            return static_file(filename, root=self.staticDir)


        @self.app.get('/<project>/getProjectStatistics')
        def get_project_statistics(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            stats = self.middleware.getProjectStatistics(project)
            return { 'statistics': stats }


        @self.app.get('/<project>/getLabelclassStatistics')
        def get_labelclass_statistics(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            stats = self.middleware.getLabelclassStatistics(project)
            return { 'statistics': stats }


        @self.app.post('/<project>/getUserStatistics')
        def get_user_statistics(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            params = request.json
            usernames_eval = params['users_eval']
            username_target = params['user_target']
            if 'threshold' in params:
                threshold = params['threshold']
            else:
                threshold = None
            if 'goldenQuestionsOnly' in params:
                goldenQuestionsOnly = params['goldenQuestionsOnly']
            else:
                goldenQuestionsOnly = False

            stats = self.middleware.getUserStatistics(project, usernames_eval, username_target, threshold, goldenQuestionsOnly)

            return { 'result': stats }