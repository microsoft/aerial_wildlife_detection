'''
    Middleware for administrative functionalities
    of AIDE.

    2020 Benjamin Kellenberger
'''

import os
import requests
from psycopg2 import sql
from constants.version import AIDE_VERSION
from modules.Database.app import Database
from util.helpers import is_localhost


class AdminMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    def getServiceDetails(self, warn_error=False):
        '''
            Queries the indicated AIController and FileServer
            modules for availability and their version. Returns
            metadata about the setup of AIDE accordingly.
            Raises an Exception if not running on the main host.
            If "warn_error" is True, a warning statement is printed
            to the command line if the version of AIDE on the attached
            AIController and/or FileServer is not the same as on the
            host, or if the servers cannot be contacted.
        '''
        # check if running on the main host
        modules = os.environ['AIDE_MODULES'].strip().split(',')
        modules = set([m.strip() for m in modules])
        if not 'LabelUI' in modules:
            # not running on main host
            raise Exception('Not a main host; cannot query service details.')

        aic_uri = self.config.getProperty('Server', 'aiController_uri', type=str, fallback=None)
        fs_uri = self.config.getProperty('Server', 'dataServer_uri', type=str, fallback=None)

        if not is_localhost(aic_uri):
            # AIController runs on a different machine; poll for version of AIDE
            try:
                aic_response = requests.get(os.path.join(aic_uri, 'version'))
                aic_version = aic_response.text
                if warn_error and aic_version != AIDE_VERSION:
                    print('WARNING: AIDE version of connected AIController differs from main host.')
                    print(f'\tAIController URI: {aic_uri}')
                    print(f'\tAIController AIDE version:    {aic_version}')
                    print(f'\tAIDE version on this machine: {AIDE_VERSION}')
            except Exception as e:
                if warn_error:
                    print(f'WARNING: error connecting to AIController (message: "{str(e)}").')
                aic_version = None
        else:
            aic_version = AIDE_VERSION
        if not is_localhost(fs_uri):
            # same for the file server
            try:
                fs_response = requests.get(os.path.join(fs_uri, 'version'))
                fs_version = fs_response.text
                if warn_error and aic_version != AIDE_VERSION:
                    print('WARNING: AIDE version of connected FileServer differs from main host.')
                    print(f'\tFileServer URI: {fs_uri}')
                    print(f'\tFileServer AIDE version:       {fs_version}')
                    print(f'\tAIDE version on this machine: {AIDE_VERSION}')
            except Exception as e:
                if warn_error:
                    print(f'WARNING: error connecting to FileServer (message: "{str(e)}").')
                fs_version = None
        else:
            fs_version = AIDE_VERSION

        # query database
        dbVersion = self.dbConnector.execute('SHOW server_version;', None, 1)[0]['server_version']
        try:
            dbVersion = dbVersion.split(' ')[0].strip()
        except:
            pass
        dbInfo = self.dbConnector.execute('SELECT version() AS version;', None, 1)[0]['version']

        return {
                'aide_version': AIDE_VERSION,
                'AIController': {
                    'uri': aic_uri,
                    'aide_version': aic_version
                },
                'FileServer': {
                    'uri': fs_uri,
                    'aide_version': fs_version
                },
                'Database': {
                    'version': dbVersion,
                    'details': dbInfo
                }
            }


    def getProjectDetails(self):
        '''
            Returns projects and statistics about them
            (number of images, disk usage, etc.).
        '''

        # get all projects
        projects = {}
        response = self.dbConnector.execute('''
                SELECT shortname, name, owner, annotationtype, predictiontype,
                    ispublic, demomode, ai_model_enabled, interface_enabled,
                    COUNT(username) AS num_users
                FROM aide_admin.project AS p
                JOIN aide_admin.authentication AS auth
                ON p.shortname = auth.project
                GROUP BY shortname
            ''', None, 'all')

        for r in response:
            projDef = {}
            for key in r.keys():
                if key != 'shortname':
                    projDef[key] = r[key]
            projects[r['shortname']] = projDef
        
        # get statistics (number of annotations, predictions, prediction models, etc.)
        for project in projects.keys():
            stats = self.dbConnector.execute(sql.SQL('''
                    SELECT COUNT(*) AS count
                    FROM {id_img}
                    UNION ALL
                    SELECT COUNT(*)
                    FROM {id_anno}
                    UNION ALL
                    SELECT COUNT(*)
                    FROM {id_pred}
                    UNION ALL
                    SELECT SUM(viewcount)
                    FROM {id_iu}
                    UNION ALL
                    SELECT COUNT(*)
                    FROM {id_cnnstate}
                ''').format(
                    id_img=sql.Identifier(project, 'image'),
                    id_anno=sql.Identifier(project, 'annotation'),
                    id_pred=sql.Identifier(project, 'prediction'),
                    id_iu=sql.Identifier(project, 'image_user'),
                    id_cnnstate=sql.Identifier(project, 'cnnstate')
                ), None, 'all')
            projects[project]['num_img'] = stats[0]['count']
            projects[project]['num_anno'] = stats[0]['count']
            projects[project]['num_pred'] = stats[0]['count']
            projects[project]['total_viewcount'] = stats[0]['count']
            projects[project]['num_cnnstates'] = stats[0]['count']

            # time statistics (last viewed)
            stats = self.dbConnector.execute(sql.SQL('''
                SELECT MIN(first_checked) AS first_checked,
                    MAX(last_checked) AS last_checked
                FROM {id_iu};
            ''').format(
                id_iu=sql.Identifier(project, 'image_user')
            ), None, 1)
            try:
                projects[project]['first_checked'] = stats[0]['first_checked'].timestamp()
            except:
                projects[project]['first_checked'] = None
            try:
                projects[project]['last_checked'] = stats[0]['last_checked'].timestamp()
            except:
                projects[project]['last_checked'] = None

        return projects


    def getUserDetails(self):
        '''
            Returns details about the user (name, number of
            enrolled projects, last activity, etc.).
        '''
        users = {}
        userData = self.dbConnector.execute('''
                SELECT name, email, isSuperUser, canCreateProjects,
                    last_login, project, isAdmin, admitted_until, blocked_until
                FROM aide_admin.user AS u
                LEFT OUTER JOIN aide_admin.authentication AS auth
                ON u.name = auth.username
            ''', None, 'all')
        for ud in userData:
            if not ud['name'] in users:
                users[ud['name']] = {
                    'email': ud['email'],
                    'canCreateProjects': ud['cancreateprojects'],
                    'isSuperUser': ud['issuperuser'],
                    'last_login': (None if ud['last_login'] is None else ud['last_login'].timestamp()),
                    'enrolled_projects': {}
                }
                if ud['project'] is not None:
                    users[ud['name']]['enrolled_projects'][ud['project']] = {
                        'admitted_until': ud['admitted_until'],
                        'blocked_until': ud['blocked_until']
                    }
        return users