'''
    Bottle routings for labeling statistics of project,
    including per-user analyses and progress.

    2019 Benjamin Kellenberger
'''

from bottle import static_file, abort
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

        @self.app.route('/<project>/statistics/static/<filename:re:.*>')
        def send_static(project, filename):
            return static_file(filename, root=self.staticDir)


        @self.app.get('/<project>/getProjectStatistics')
        def get_project_statistics(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            stats = self.middleware.getProjectStatistics(project)
            return { 'statistics': stats }