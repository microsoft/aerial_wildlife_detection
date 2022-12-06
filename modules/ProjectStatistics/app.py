'''
    Bottle routings for labeling statistics of project,
    including per-user analyses and progress.

    2019-21 Benjamin Kellenberger
'''

import html
from bottle import request, static_file, abort
from .backend.middleware import ProjectStatisticsMiddleware
from util.helpers import parse_boolean


class ProjectStatistics:

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app
        self.staticDir = 'modules/ProjectStatistics/static'
        self.middleware = ProjectStatisticsMiddleware(config, dbConnector)

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


        @self.app.post('/<project>/getPerformanceStatistics')
        def get_user_statistics(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')

            try:
                params = request.json
                entities_eval = params['entities_eval']
                entity_target = params['entity_target']
                entityType = params['entity_type']
                if 'threshold' in params:
                    threshold = params['threshold']
                else:
                    threshold = None
                if 'goldenQuestionsOnly' in params:
                    goldenQuestionsOnly = params['goldenQuestionsOnly']
                else:
                    goldenQuestionsOnly = False

                stats = self.middleware.getPerformanceStatistics(project, entities_eval, entity_target, entityType, threshold, goldenQuestionsOnly)

                return {
                    'status': 0,
                    'result': stats
                }

            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }


        
        @self.app.post('/<project>/getUserAnnotationSpeeds')
        def get_user_annotation_speeds(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            userList = request.json['users']
            if 'goldenQuestionsOnly' in request.json:
                goldenQuestionsOnly = request.json['goldenQuestionsOnly']
            else:
                goldenQuestionsOnly = False
            
            stats = self.middleware.getUserAnnotationSpeeds(project, userList, goldenQuestionsOnly)
            
            return { 'result': stats }


        @self.app.get('/<project>/getUserFinished')
        def get_user_finished(project):
            if not self.loginCheck(project=project):
                abort(401, 'forbidden')
            
            try:
                username = html.escape(request.get_cookie('username'))
                done = self.middleware.getUserFinished(project, username)
            except Exception:
                done = False
            return {'finished': done}


        @self.app.get('/<project>/getTimeActivity')
        def get_time_activity(project):
            if not self.loginCheck(project=project, admin=True):
                abort(401, 'forbidden')
            
            try:
                try:
                    type = request.query['type']
                except Exception:
                    type = 'image'
                try:
                    numDaysMax = request.query['num_days']
                except Exception:
                    numDaysMax = 31
                try:
                    perUser = parse_boolean(request.query['per_user'])
                except Exception:
                    perUser = False
                stats = self.middleware.getTimeActivity(project, type, numDaysMax, perUser)
                return {'result': stats}
            except Exception as e:
                abort(401, str(e))