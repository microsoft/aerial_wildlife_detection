'''
    Main Bottle and routings for the LabelUI web frontend.

    2019 Benjamin Kellenberger
'''

import os
import cgi
import bottle
from bottle import request, response, static_file, redirect, abort, SimpleTemplate
from .backend.middleware import DBMiddleware


#TODO
bottle.BaseRequest.MEMFILE_MAX = 1024**3

class LabelUI():

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = 'modules/LabelUI/static'
        self.middleware = DBMiddleware(config)

        self.demoMode = config.getProperty('Project', 'demoMode', type=bool, fallback=False)    #TODO: project-specific

        self.login_check = None

        self._initBottle()

    
    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return True if self.demoMode or self.login_check is None else self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        if not self.demoMode:
            self.login_check = loginCheckFun


    def __redirect_login_page(self):
        response = bottle.response
        response.status = 303
        response.set_header('Location', '/login')
        return response


    def _initBottle(self):

        ''' static routings '''
        # @self.app.route('/')
        # def index():
        #     # redirect to interface if logged in    TODO
        #     if self.loginCheck():
        #         return redirect('/interface')
        #     else:
        #         return static_file('index.html', root=os.path.join(self.staticDir, 'templates'))

        # @self.app.route('/favicon.ico')
        # def favicon():
        #     return static_file('favicon.ico', root=os.path.join(self.staticDir, 'img'))

        # @self.app.route('/about')
        # def about():
        #     return static_file('about.html', root=os.path.join(self.staticDir, 'templates'))

        @self.app.route('/<project>')
        @self.app.route('/<project>/')
        def project_page(project):
            #TODO: show advanced project controls
            return redirect('/' + project + '/interface')


        with open(os.path.abspath(os.path.join('modules/LabelUI/static/templates/interface.html')), 'r') as f:
            self.interface_template = SimpleTemplate(f.read())

        @self.app.route('/<project>/interface')
        def interface(project):
            
            # check if user logged in
            if not self.loginCheck(project=project):
                return self.__redirect_login_page()

            # get project data (and check if project exists)
            projectData = self.middleware.getProjectInfo(project)
            if projectData is None:
                return self.__redirect_login_page()
            
            # check if user authenticated for project
            if not self.loginCheck(project=project, extend_session=True):
                return self.__redirect_login_page()
            
            # render interface template
            username = 'Demo mode' if self.demoMode else cgi.escape(request.get_cookie('username'))
            return self.interface_template.render(username=username,
                projectShortname=project,
                projectTitle=projectData['projectName'], projectDescr=projectData['projectDescription'])


        # @self.app.route('/static/<filename:re:.*>')
        # def send_static(filename):
        #     return static_file(filename, root=self.staticDir)
        @self.app.route('/<project>/static/<filename:re:.*>')
        @self.app.route('/<project>/interface/static/<filename:re:.*>')
        def send_static_proj(project, filename):
            return static_file(filename, root=self.staticDir)


        @self.app.route('/backdrops/<filename:re:.*>')
        def send_backdrop_image(filename):
            try:
                return static_file(filename, root=self.middleware.projectSettings['backdrops']['basePath'])
            except:
                abort(404, 'backdrop not found')


        @self.app.route('/<project>/backdrops/<filename:re:.*>')
        def send_backdrop_image_proj(project, filename):
            try:
                return static_file(filename, root=self.middleware.projectSettings['backdrops']['basePath'])
            except:
                abort(404, 'backdrop not found')


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
                classDefs = {
                    'classes': self.middleware.getClassDefinitions(project)
                }
                return classDefs
            else:
                abort(401, 'not logged in')


        @self.app.post('/<project>/getImages')
        def get_images(project):
            if self.loginCheck(project=project):
                username = cgi.escape(request.get_cookie('username'))
                dataIDs = request.json['imageIDs']
                json = self.middleware.getBatch_fixed(project, username, dataIDs)
                return json
            else:
                abort(401, 'not logged in')


        @self.app.get('/<project>/getLatestImages')
        def get_latest_images(project):
            if self.loginCheck(project=project):
                username = cgi.escape(request.get_cookie('username'))
                try:
                    limit = int(request.query['limit'])
                except:
                    limit = None
                try:
                    order = int(request.query['order'])
                except:
                    order = 'unlabeled'
                try:
                    subset = int(request.query['subset'])
                except:
                    subset = 'default'  
                json = self.middleware.getBatch_auto(project=project, username=username, order=order, subset=subset, limit=limit)

                return json
            else:
                abort(401, 'not logged in')


        @self.app.post('/<project>/getImages_timestamp')
        def get_images_timestamp(project):
            if self.demoMode:
                return { 'status': 'not allowed in demo mode' }

            username = cgi.escape(request.get_cookie('username'))

            # check if user requests to see other user names; only permitted if admin
            # also, by default we limit labels to the current user,
            # even for admins, to provide a consistent experience.
            try:
                users = request.json['users']
                if not len(users):
                    users = [username]
            except:
                users = [username]

            if not self.loginCheck(project=project, admin=True):
                # user no admin: can only query their own labels
                users = [username]
            
            elif not self.loginCheck(project=project):
                # not logged in, resp. not authorized for project
                abort(401, 'unauthorized')
                
            try:
                minTimestamp = request.json['minTimestamp']
            except:
                minTimestamp = None
            try:
                maxTimestamp = request.json['maxTimestamp']
            except:
                maxTimestamp = None
            try:
                skipEmpty = request.json['skipEmpty']
            except:
                skipEmpty = False
            try:
                limit = request.json['limit']
            except:
                limit = None

            # query and return
            json = self.middleware.getBatch_timeRange(project, minTimestamp, maxTimestamp, users, skipEmpty, limit)
            return json


        @self.app.post('/<project>/getTimeRange')
        def get_time_range(project):
            if self.demoMode:
                return { 'status': 'not allowed in demo mode' }

            username = cgi.escape(request.get_cookie('username'))

            # check if user requests to see other user names; only permitted if admin
            try:
                users = request.json['users']
            except:
                # no users provided; restrict to current account
                users = [username]
            

            if not self.loginCheck(project=project, admin=True):
                # user no admin: can only query their own labels
                users = [username]
            
            elif not self.loginCheck(project=project):
                # not logged in, resp. not authorized for this project
                abort(401, 'unauthorized')

            try:
                skipEmpty = request.json['skipEmpty']
            except:
                skipEmpty = False

            # query and return
            json = self.middleware.get_timeRange(project, users, skipEmpty)
            return json


        @self.app.post('/<project>/submitAnnotations')
        def submit_annotations(project):
            if self.demoMode:
                return { 'status': 'not allowed in demo mode' }
            
            if self.loginCheck(project=project):
                # parse
                try:
                    username = cgi.escape(request.get_cookie('username'))
                    if username is None:
                        # this should never happen, since we are performing a login check
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