'''
    Web Map Service (Mapserver) middleware.

    2023 Benjamin Kellenberger
'''

from typing import Tuple, Iterable
import time
import json
from psycopg2 import sql

from modules.Database.app import Database
from util.configDef import Config
from util import helpers, geospatial

from . import service_renderers

class MapserverMiddleware:
    '''
        Mapserver middleware
    '''

    # seconds to wait until project metadata is re-queried from database
    #TODO: make global option
    METADATA_QUERY_INTERVAL = 150

    def __init__(self, config: Config, db_connector: Database) -> None:
        self.config = config
        self.db_connector = db_connector

        self.services = {}
        self._init_services()

        self.project_meta = {}
        self.get_project_meta()

        self.user_access = {}
        self.get_user_access()


    def _init_services(self) -> None:
        for renderer in service_renderers.__all__:
            renderer_class = getattr(service_renderers, renderer)
            self.services[renderer_class.SERVICE_NAME] = renderer_class(self.config,
                                                                        self.db_connector)

    def get_user_access(self, username: str=None, projects: Iterable=None) -> dict:
        '''
            Returns privileges for one or more given "projects" and "username" about access,
            including:

            - whether the user is registered in each project
            - the user's role (member, admin) therein

            Results determine whether the user can access a project, and which layers they can if
            so. Caches information for faster retrieval. Always returns a response, even if the
            username and/or project do not exist.

            If "username" is None, all registered users in AIDE will be queried and stored. If
            "projects" is None, all registered projects will be queried.

            Only returns valid information if at least "username" is provided.
        '''
        now = time.time()
        default_response = {
            'is_member': False,
            'is_admin': False
        }
        if projects is not None and isinstance(projects, str):
            projects = (projects,)
        if username is None or username not in self.user_access or \
            now - self.user_access[username].get('last_queried', 0) > self.METADATA_QUERY_INTERVAL:
            user_sql, project_sql = '', ''
            query_args = []
            if username is not None:
                query_args.append(username)
                user_sql = 'WHERE username = %s'
            if projects is not None:
                query_args.extend(projects)
                project_sql = f'''{'AND' if len(user_sql)>0 else 'WHERE'} username IN
                    ({','.join(['%s' for _ in range(len(projects))])})'''
            result = self.db_connector.execute(sql.SQL('''
                SELECT username, project, isadmin, admitted_until, blocked_until
                FROM {id_auth}
                {user_sql} {project_sql};
            ''').format(
                id_auth=sql.Identifier('aide_admin', 'authentication'),
                user_sql=sql.SQL(user_sql),
                project_sql=sql.SQL(project_sql)
            ), query_args, 'all')
            for row in result:
                user = row['username']
                if 'projects' not in self.user_access.get(user, {}):
                    self.user_access[user] = {
                        'last_queried': now,
                        'projects': {}
                    }
                project = row['project']
                if (row['admitted_until'] is None or (row['admitted_until'] >= now)) and \
                    (row['blocked_until'] is None or (row['blocked_until'] < now)):
                    # user has access to project
                    self.user_access[row['username']]['projects'][project] = {
                        'is_member': True,
                        'is_admin': row['isadmin']
                    }
        # return values
        if username is not None:
            acl = self.user_access.get(username, {}).get('projects', {})
            if projects is not None:
                return dict([key, acl.get(key, default_response)] for key in projects)
            return acl
        return default_response


    def get_project_meta(self, project: str=None) -> dict:
        '''
            Returns project-specific metadata, including:
            - annotation type *
            - prediction type *
            - band config *
            - render config
            - label classes (with names, indices, and colors)
            - SRID *
            - registered users
            - whether project runs in demo mode
            Caches information for faster retrieval.
            If "project" is None, all projects will be queried.
        '''
        now = time.time()
        if project is None or \
            project not in self.project_meta or \
            now - self.project_meta[project]['last_queried'] > self.METADATA_QUERY_INTERVAL:
            # (re-) query from database
            proj_str = '' if project is None else 'WHERE shortname = %s'
            query_args = None if project is None else (project,)

            meta = self.db_connector.execute(sql.SQL('''
                SELECT srid, shortname, band_config, render_config,
                    annotationtype, predictiontype,
                    proj.demomode,
                    auth.username, auth.isadmin, auth.admitted_until, auth.blocked_until
                FROM {id_proj} AS proj
                FULL OUTER JOIN geometry_columns AS geom
                ON proj.shortname = geom.f_table_schema
                JOIN {id_auth} AS auth
                ON proj.shortname = auth.project
                {proj_str};
            ''').format(
                id_proj=sql.Identifier('aide_admin', 'project'),
                id_auth=sql.Identifier('aide_admin', 'authentication'),
                proj_str=sql.SQL(proj_str)
            ), query_args, 'all')
            if meta is None or len(meta) == 0:
                return None
            projects_queried = set()
            for item in meta:
                proj_name = item['shortname']
                if proj_name not in self.project_meta:
                    self.project_meta[proj_name] = {
                        'last_queried': now,
                        'annotation_type': item['annotationtype'].lower(),
                        'prediction_type': item['predictiontype'].lower(),
                        'band_config': json.loads(item['band_config']),
                        'render_config': json.loads(item['render_config']),
                        'srid': item['srid'],
                        'extent': geospatial.get_project_extent(self.db_connector, proj_name),
                        'demo_mode': item['demomode'],
                        'users': {}
                    }
                # check member list
                if (item['admitted_until'] is None or (item['admitted_until'] >= now)) and \
                    (item['blocked_until'] is None or (item['blocked_until'] < now)):
                    # user has access to project
                    user_name = item['username']
                    self.project_meta[proj_name]['users'][user_name] = {
                        'name': user_name,
                        'is_admin': item['isadmin']
                    }
                if proj_name not in projects_queried:
                    # get label class info
                    lc_meta = self.db_connector.execute(
                        sql.SQL('SELECT * FROM {};').format(
                            sql.Identifier(proj_name, 'labelclass')),
                            None, 'all')
                    label_classes = [
                        {
                            'name': item['name'],
                            'idx': item['idx'],
                            'color': helpers.hexToRGB(item.get('color', '#000000'))
                        }
                        for item in lc_meta
                    ]
                    self.project_meta[proj_name]['label_classes'] = label_classes
                    projects_queried.add(proj_name)
        return self.project_meta.get(project, None)


    def _get_access_control(self, project: str, username: str=None) -> dict:
        '''
            Returns True if a given user has access to a project, else False.
        '''
        access_control = self.get_user_access(username, project)
        proj_meta = self.get_project_meta(project)
        if proj_meta is None:
            return {
                'has_access': False,
                'is_admin': False,
                'project_meta': {}
            }
        if access_control.get('is_member', True) is False:
            # username not found
            return {
                'has_access': proj_meta.get('demo_mode', False),
                'is_admin': False,
                'project_meta': {}
            }
        if not access_control.get('is_admin', False):
            # user is no admin; drop all names except their own
            proj_meta['users'] = {username: proj_meta['users'][username]}
        return {
            'has_access': proj_meta.get('demo_mode', False) or \
                            access_control.get(project, {}).get('is_member', False),
            'is_admin': access_control.get('is_admin', False),
            'project_meta': proj_meta
        }


    def service(self, service: str,
                        request: str,
                        request_params: dict,
                        projects: Iterable,
                        username: str,
                        base_url: str) -> Tuple[object, dict]:
        '''
            Mapserver service implementation.
        '''
        service = service.lower()
        if service not in self.services:
            raise Exception(f'Unsupported service "{service}"')

        # gather project metadata, also checking user privileges
        if projects is None:
            user_access = self.get_user_access(username)
            projects = tuple(key for key, meta in user_access.items()
                                if meta.get('is_member', False))
        elif isinstance(projects, str):
            projects = (projects,)

        project_meta = {}
        for project in projects:
            access_control = self._get_access_control(project, username)
            if not access_control['has_access']:
                continue
            project_meta[project] = access_control['project_meta']

        return self.services[service](request,
                                        project_meta,
                                        base_url,
                                        request_params)


    def __call__(self, service: str,
                        request: str,
                        request_params: dict,
                        projects: Iterable,
                        username: str,
                        base_url: str) -> Tuple[object, dict]:
        '''
            Alias for Mapserver service implementation.
        '''
        return self.service(service,
                            request,
                            request_params,
                            projects,
                            username,
                            base_url)
