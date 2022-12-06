'''
    Main Bottle and routings for the LabelUI web frontend.

    2019-21 Benjamin Kellenberger
'''

import os
import html
from uuid import UUID
import bottle
from bottle import request, redirect, abort, SimpleTemplate
from constants.version import AIDE_VERSION
from .backend.middleware import DBMiddleware
from util.helpers import LogDecorator, parse_boolean



class LabelUI():

    def __init__(self, config, app, dbConnector, verbose_start=False):
        self.config = config
        self.app = app
        self.staticDir = 'modules/LabelUI/static'

        if verbose_start:
            print('LabelUI'.ljust(LogDecorator.get_ljust_offset()), end='')
        
        try:
            self.middleware = DBMiddleware(config, dbConnector)
            self.login_check = None

            self._initBottle()
        except Exception as e:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not launch LabelUI (message: "{str(e)}").')

        if verbose_start:
            LogDecorator.print_status('ok')

    
    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def __redirect_login_page(self, redirect=None):
        location = '/login'
        if redirect is not None:
            location += '?redirect=' + redirect
        response = bottle.response
        response.status = 303
        response.set_header('Location', location)
        return response


    def __redirect_project_page(self, project):
        response = bottle.response
        response.status = 303
        response.set_header('Location', '/')    #TODO: add project once loopback is resolved and project page initiated
        return response


    def _initBottle(self):

        ''' static routings '''
        # @self.app.route('/<project>')
        # @self.app.route('/<project>/')
        # def project_page(project):
        #     #TODO: show advanced project controls
        #     return redirect('/' + project + '/interface')


        with open(os.path.abspath(os.path.join('modules/LabelUI/static/templates/interface.html')), 'r') as f:
            self.interface_template = SimpleTemplate(f.read())

        @self.app.route('/<project>/interface')
        def interface(project):
            
            # check if user logged in
            if not self.loginCheck(project=project):
                return self.__redirect_login_page(project + '/interface')
            
            # check if user is enrolled in project; redirect if not
            if not self.loginCheck(project=project):
                return redirect('/' + project)      #TODO: verify

            # get project data (and check if project exists)
            projectData = self.middleware.getProjectInfo(project)
            if projectData is None:
                return self.__redirect_login_page('/')
            
            # check if user authenticated for project
            if not self.loginCheck(project=project, extend_session=True):
                return self.__redirect_login_page(project + '/interface')
            
            # redirect to project page if interface not enabled
            if not projectData['interface_enabled']:
                return self.__redirect_project_page(project)

            # render interface template
            try:
                username = html.escape(request.get_cookie('username'))
            except Exception:
                username = ''
            return self.interface_template.render(username=username,
                version=AIDE_VERSION,
                projectShortname=project,
                projectTitle=projectData['projectName'], projectDescr=projectData['projectDescription'])
        

        @self.app.get('/<project>/getProjectInfo')
        def get_project_info(project):
            # minimum info (name, description) that can be viewed without logging in
            return {'info': self.middleware.getProjectInfo(project)}


        @self.app.get('/<project>/getProjectSettings')
        def get_project_settings(project):
            if self.loginCheck(project=project):
                settings = {
                    'settings': self.middleware.getProjectSettings(project)
                }
                return settings
            else:
                abort(401, 'not logged in')


        @self.app.get('/<project>/getClassDefinitions')
        def get_class_definitions(project):
            if self.loginCheck(project=project):
                try:
                    showHidden = parse_boolean(request.params['show_hidden'])
                except Exception:
                    showHidden = False
                classDefs = {
                    'classes': self.middleware.getClassDefinitions(project, showHidden)
                }
                return classDefs
            else:
                abort(401, 'not logged in')


        @self.app.post('/<project>/getImages')
        def get_images(project):
            if self.loginCheck(project=project):
                hideGoldenQuestionInfo = True
                if self.loginCheck(project=project, admin=True):
                    hideGoldenQuestionInfo = False

                try:
                    username = html.escape(request.get_cookie('username'))
                except Exception:
                    username = ''
                dataIDs = request.json['imageIDs']
                json = self.middleware.getBatch_fixed(project, username, dataIDs, hideGoldenQuestionInfo)
                return json
            else:
                abort(401, 'not logged in')


        @self.app.get('/<project>/getLatestImages')
        def get_latest_images(project):
            if self.loginCheck(project=project):
                hideGoldenQuestionInfo = True
                if self.loginCheck(project=project, admin=True):
                    hideGoldenQuestionInfo = False

                try:
                    username = html.escape(request.get_cookie('username'))
                except Exception:
                    username = ''
                try:
                    limit = int(request.query['limit'])
                except Exception:
                    limit = None
                try:
                    order = int(request.query['order'])
                except Exception:
                    order = 'unlabeled'
                try:
                    subset = int(request.query['subset'])
                except Exception:
                    subset = 'default'  
                json = self.middleware.getBatch_auto(project=project, username=username, order=order, subset=subset, limit=limit, hideGoldenQuestionInfo=hideGoldenQuestionInfo)

                return json
            else:
                abort(401, 'not logged in')


        @self.app.post('/<project>/getImages_timestamp')
        def get_images_timestamp(project):
            if not self.loginCheck(project=project):
                abort(401, 'unauthorized')

            # check if user requests to see other user names; only permitted if admin
            # also, by default we limit labels to the current user,
            # even for admins, to provide a consistent experience.
            username = html.escape(request.get_cookie('username'))
            try:
                users = request.json['users']
                if not len(users):
                    users = [username]
            except Exception:
                users = [username]

            if not self.loginCheck(project=project, admin=True):
                # user no admin: can only query their own labels
                users = [username]
                hideGoldenQuestionInfo = True
            
            elif not self.loginCheck(project=project):
                # not logged in, resp. not authorized for project
                abort(401, 'unauthorized')

            else:
                hideGoldenQuestionInfo = False
            
            minTimestamp = request.json.get('minTimestamp', None)
            maxTimestamp = request.json.get('maxTimestamp', None)
            skipEmpty = request.json.get('skipEmpty', False)
            limit = request.json.get('limit', None)
            goldenQuestionsOnly = request.json.get('goldenQuestionsOnly', False)
            lastImageUUID = request.json.get('lastImageUUID', None)

            # query and return
            json = self.middleware.getBatch_timeRange(project, minTimestamp, maxTimestamp, users, skipEmpty, limit, goldenQuestionsOnly, hideGoldenQuestionInfo, lastImageUUID)
            return json


        @self.app.post('/<project>/getTimeRange')
        def get_time_range(project):
            if not self.loginCheck(project=project):
                abort(401, 'unauthorized')

            username = html.escape(request.get_cookie('username'))

            # check if user requests to see other user names; only permitted if admin
            try:
                users = request.json['users']
            except Exception:
                # no users provided; restrict to current account
                users = [username]
            

            if not self.loginCheck(project=project, admin=True):
                # user no admin: can only query their own labels
                users = [username]

            try:
                skipEmpty = request.json['skipEmpty']
            except Exception:
                skipEmpty = False
            try:
                goldenQuestionsOnly = request.json['goldenQuestionsOnly']
            except Exception:
                goldenQuestionsOnly = False

            # query and return
            json = self.middleware.get_timeRange(project, users, skipEmpty, goldenQuestionsOnly)
            return json


        @self.app.get('/<project>/getSampleData')
        @self.app.post('/<project>/getSampleData')
        def get_sample_data(project):
            if self.loginCheck(project=project):
                
                json = self.middleware.get_sampleData(project)
                return json
            else:
                abort(401, 'not logged in')


        @self.app.post('/<project>/submitAnnotations')
        def submit_annotations(project):
            if self.loginCheck(project=project):
                # parse
                try:
                    username = html.escape(request.get_cookie('username'))
                    if username is None:
                        # 100% failsafety for projects in demo mode
                        raise Exception('no username provided')
                    submission = request.json
                    status = self.middleware.submitAnnotations(project, username, submission)
                    return { 'status': status }
                except Exception as e:
                    return {
                        'status': 1,
                        'message': str(e)
                    }
            else:
                abort(401, 'not logged in')


        @self.app.get('/<project>/getGoldenQuestions')
        def get_golden_questions(project):
            if not self.loginCheck(project=project, admin=True):
                abort(403, 'forbidden')
            try:
                return self.middleware.getGoldenQuestions(project)
            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }

        
        @self.app.post('/<project>/setGoldenQuestions')
        def set_golden_questions(project):
            if self.loginCheck(project=project, admin=True):
                # parse and check validity of submissions
                submissions = request.json
                try:
                    submissions = submissions['goldenQuestions']
                    if not isinstance(submissions, dict):
                        abort(400, 'malformed submissions')
                    submissions_ = []
                    for key in submissions.keys():
                        submissions_.append(tuple((UUID(key), bool(submissions[key]),)))
                except Exception:
                    abort(400, 'malformed submissions')

                response = self.middleware.setGoldenQuestions(project, tuple(submissions_))

                return response

            else:
                abort(403, 'forbidden')


        @self.app.get('/<project>/getBookmarks')
        def get_bookmarks(project):
            if not self.loginCheck(project=project):
                abort(403, 'forbidden')
            try:
                username = html.escape(request.get_cookie('username'))
                if username is None:
                    abort(403, 'forbidden')
                return self.middleware.getBookmarks(project, username)
            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }


        @self.app.post('/<project>/setBookmark')
        def set_bookmark(project):
            if not self.loginCheck(project=project):
                abort(403, 'forbidden')
            
            try:
                username = html.escape(request.get_cookie('username'))
                if username is None:
                    # 100% failsafety for projects in demo mode
                    raise Exception('no username provided')

                bookmarks = request.json['bookmarks']
                response = self.middleware.setBookmark(project, username, bookmarks)
                return response
            except Exception as e:
                return {
                    'status': 1,
                    'message': str(e)
                }