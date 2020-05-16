'''
    This module handles everything about the
    setup and monitoring of AIDE (i.e., super
    user functionality).

    2020 Benjamin Kellenberger
'''

import os
import html
from bottle import SimpleTemplate, request, redirect, abort
from constants.version import AIDE_VERSION
from .backend.middleware import AdminMiddleware



class AIDEAdmin:

    def __init__(self, config, app):
        self.config = config
        self.app = app
        self.staticDir = 'modules/AIDEAdmin/static'
        self.login_check = None

        self.middleware = AdminMiddleware(config)

        # ping connected AIController, FileServer, etc. servers and check version
        try:
            self.middleware.getServiceDetails(True)
        except Exception as e:
            pass

        self._initBottle()
    

    def loginCheck(self, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        return self.login_check(project, admin, superuser, canCreateProjects, extend_session)


    def addLoginCheckFun(self, loginCheckFun):
        self.login_check = loginCheckFun


    def _initBottle(self):

        # read AIDE admin templates
        with open(os.path.abspath(os.path.join(self.staticDir, 'templates/aideAdmin.html')), 'r') as f:
            self.adminTemplate = SimpleTemplate(f.read())

        self.panelTemplates = {}
        panelNames = os.listdir(os.path.join(self.staticDir, 'templates/panels'))
        for pn in panelNames:
            pnName, ext = os.path.splitext(pn)
            if ext.lower().startswith('.htm'):
                with open(os.path.join(self.staticDir, 'templates/panels', pn), 'r') as f:
                    self.panelTemplates[pnName] = SimpleTemplate(f.read())


        @self.app.route('/admin/config/panels/<panel>')
        def send_static_panel(panel):
            if self.loginCheck(superuser=True):
                try:
                    return self.panelTemplates[panel].render(
                        version=AIDE_VERSION
                    )
                except:
                    abort(404, 'not found')
            else:
                abort(401, 'forbidden')


        @self.app.route('/admin')
        def send_aide_admin_page():
            if not self.loginCheck(superuser=True):
                return redirect('/')

            # render configuration template
            try:
                username = html.escape(request.get_cookie('username'))
            except:
                return redirect('/')

            return self.adminTemplate.render(
                    version=AIDE_VERSION,
                    username=username)


        @self.app.route('/getServiceDetails')
        def get_service_details():
            try:
                if not self.loginCheck(superuser=True):
                    return redirect('/')
                return {'details': self.middleware.getServiceDetails()}
            except:
                abort(404, 'not found')


        