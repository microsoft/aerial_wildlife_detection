'''
    Middleware layer between the project configuration front-end
    and the database.

    2019-22 Benjamin Kellenberger
'''

import os
import re
import secrets
import json
import uuid
from datetime import datetime
from collections.abc import Iterable
import requests
from psycopg2 import sql
from bottle import request

from modules.DataAdministration.backend import celery_interface as fileServer_interface
from modules.TaskCoordinator.backend.middleware import TaskCoordinatorMiddleware
from .db_fields import Fields_annotation, Fields_prediction
from util import helpers


class ProjectConfigMiddleware:

    # prohibited project shortnames
    PROHIBITED_SHORTNAMES = [
        'con',  # for MS Windows
        'prn',
        'aux',
        'nul',
        'com1',
        'com2',
        'com3',
        'com4',
        'com5',
        'com6',
        'com7',
        'com8',
        'com9',
        'lpt1',
        'lpt2',
        'lpt3',
        'lpt4',
        'lpt5',
        'lpt6',
        'lpt7',
        'lpt8',
        'lpt9'
    ]

    # prohibited project names (both as a whole and for shortnames)
    PROHIBITED_NAMES = [
        '',
        'project',
        'getavailableaimodels',
        'getbackdrops',
        'verifyprojectname',
        'verifyprojectshort',
        'newproject',
        'createproject',
        'statistics',
        'static',
        'getcreateaccountunrestricted'
        'getprojects',
        'about',
        'favicon.ico',
        'logincheck',
        'logout',
        'login',
        'dologin',
        'createaccount',
        'loginscreen',
        'accountexists',
        'getauthentication',
        'getusernames',
        'docreateaccount',
        'admin',
        'getservicedetails',
        'getceleryworkerdetails',
        'getprojectdetails',
        'getuserdetails',
        'setpassword',
        'exec',
        'v',
        'getbandconfiguration'
    ]

    # prohibited name prefixes
    PROHIBITED_NAME_PREFIXES = [
        '/',
        '?',
        '&'
    ]

    # patterns that are prohibited anywhere for shortnames (replaced with underscores)
    SHORTNAME_PATTERNS_REPLACE = [
        '|',
        '?',
        '*',
        ':'    # for macOS
    ]

    # patterns that are prohibited anywhere for both short and long names (no replacement)
    PROHIBITED_STRICT = [
        '&lt;',
        '<',
        '>',
        '&gt;',
        '..',
        '/',
        '\\'
    ]

    # absolute RGB component sum distance required between colors. In principle,
    # this is only important for segmentation (due to anti-aliasing effects of
    # the HTML canvas, but we apply it everywhere anyway)
    MINIMAL_COLOR_OFFSET = 9
    
    def __init__(self, config, dbConnector):
        self.config = config
        self.dbConnector = dbConnector

        # load default UI settings
        try:
            # check if custom default styles are provided
            self.defaultUIsettings = json.load(open('config/default_ui_settings.json', 'r'))
        except:
            # resort to built-in styles
            self.defaultUIsettings = json.load(open('modules/ProjectAdministration/static/json/default_ui_settings.json', 'r'))


    @staticmethod
    def _recursive_update(dictObject, target):
        '''
            Recursively iterates over all keys and sub-keys of "dictObject"
            and its sub-dicts and copies over values from dict "target", if
            they are available.
        '''
        for key in dictObject.keys():
            if key in target:
                if isinstance(dictObject[key], dict):
                    ProjectConfigMiddleware._recursive_update(dictObject[key], target[key])
                else:
                    dictObject[key] = target[key]
    
    
    def getPlatformInfo(self, project, parameters=None):
        '''
            AIDE setup-specific platform metadata.
        '''
        # parse parameters (if provided) and compare with mutable entries
        allParams = set([
            'server_uri',
            'server_dir',
            'watch_folder_interval',
            'inference_batch_size_limit',
            'max_num_concurrent_tasks'
        ])
        if parameters is not None and parameters != '*':
            if isinstance(parameters, str):
                parameters = [parameters.lower()]
            else:
                parameters = [p.lower() for p in parameters]
            set(parameters).intersection_update(allParams)
        else:
            parameters = allParams
        parameters = list(parameters)

        # check if FileServer needs to be contacted
        serverURI = self.config.getProperty('Server', 'dataServer_uri')
        serverDir = self.config.getProperty('FileServer', 'staticfiles_dir')
        if 'server_dir' in parameters and not helpers.is_localhost(serverURI):
            # FileServer is remote instance; get info via URL query
            try:
                cookies = request.cookies.dict
                for key in cookies:
                    cookies[key] = cookies[key][0]
                fsData = requests.get(os.path.join(serverURI, 'getFileServerInfo'), cookies=cookies)
                fsData = json.loads(fsData.text)
                serverDir = fsData['staticfiles_dir']
            except Exception as e:
                print(f'WARNING: an error occurred trying to query FileServer for static files directory (message: "{str(e)}").')
                print(f'Using value provided in this instance\'s config instead ("{serverDir}").')

        response = {}
        for param in parameters:
            if param.lower() == 'server_uri':
                response[param] = os.path.join(serverURI, project, 'files')
            elif param.lower() == 'server_dir':
                response[param] = os.path.join(serverDir, project)
            elif param.lower() == 'watch_folder_interval':
                interval = self.config.getProperty('FileServer', 'watch_folder_interval', type=float, fallback=60)
                response[param] = interval
            elif param.lower() == 'inference_batch_size_limit':
                inferenceBsLimit = self.config.getProperty('AIWorker', 'inference_batch_size_limit', type=int, fallback=-1)
                response[param] = inferenceBsLimit
            elif param.lower() == 'max_num_concurrent_tasks':
                maxNumConcurrentTasksLimit = self.config.getProperty('AIWorker', 'max_num_concurrent_tasks', type=int, fallback=2)
                response[param] = maxNumConcurrentTasksLimit

        return response

    
    def getProjectImmutables(self, project):
        queryStr = 'SELECT annotationType, predictionType, demoMode FROM aide_admin.project WHERE shortname = %s;'
        result = self.dbConnector.execute(queryStr, (project,), 1)
        if result and len(result):
            return {
                'annotationType': result[0]['annotationtype'],
                'predictionType': result[0]['predictiontype']
            }
        else:
            return None


    def getProjectInfo(self, project, parameters=None):

        # parse parameters (if provided) and compare with mutable entries
        allParams = set([
            'name',
            'description',
            'ispublic',
            'secret_token',
            'demomode',
            'interface_enabled',
            'archived',
            'ui_settings',
            'segmentation_ignore_unlabeled',
            'ai_model_enabled',
            'ai_model_library',
            'ai_model_settings',
            'ai_alcriterion_library',
            'ai_alcriterion_settings',
            'numimages_autotrain',
            'minnumannoperimage',
            'maxnumimages_train',
            'inference_chunk_size',
            'max_num_concurrent_tasks',
            'watch_folder_enabled',
            'watch_folder_remove_missing_enabled',
            'band_config',
            'render_config'
        ])
        if parameters is not None and parameters != '*':
            if isinstance(parameters, str):
                parameters = [parameters.lower()]
            else:
                parameters = [p.lower() for p in parameters]
            parameters = set(parameters)
            parameters.intersection_update(allParams)
            parameters.add('archived')
            parameters = list(parameters)
        else:
            parameters = allParams
        parameters = list(parameters)
        sqlParameters = ','.join(parameters)

        queryStr = sql.SQL('''
        SELECT {} FROM aide_admin.project
        WHERE shortname = %s;
        ''').format(
            sql.SQL(sqlParameters)
        )
        result = self.dbConnector.execute(queryStr, (project,), 1)
        result = result[0]

        # assemble response
        response = {}
        for param in parameters:
            value = result[param]
            if param == 'ui_settings':
                value = json.loads(value)

                # auto-complete with defaults where missing
                value = helpers.check_args(value, self.defaultUIsettings)
            elif param == 'interface_enabled':
                value = value and not result['archived']
            elif param in ('band_config', 'render_config'):
                try:
                    value = json.loads(value)
                except:
                    value = None
            response[param] = value

        return response


    def renewSecretToken(self, project):
        '''
            Creates a new secret token, invalidating the old one.
        '''
        try:
            newToken = secrets.token_urlsafe(32)
            result = self.dbConnector.execute('''UPDATE aide_admin.project
                SET secret_token = %s
                WHERE shortname = %s;
                SELECT secret_token FROM aide_admin.project
                WHERE shortname = %s;
            ''', (newToken, project, project,), 1)
            return result[0]['secret_token']
        except:
            # this normally should not happen, since we checked for the validity of the project
            return None


    def setPermissions(self, project, userList, privileges):
        '''
            Sets project permissions for a given list of user names.
            Permissions may be set through a dict of "privileges" with
            values and include the following privilege keywords and
            value types:
                - "isAdmin": bool
                - "blocked_until": datetime or anything else for no limit
                - "admitted_until": datetime or anything else for no limit
                - "remove": bool        # removes users from project
        '''
        userList = [(u,) for u in userList]

        for p in privileges.keys():
            queryType = 'update'
            if p.lower() == 'isadmin':
                queryVal = bool(privileges[p])
            elif p.lower() in ('admitted_until', 'blocked_until'):
                try:
                    queryVal = datetime.fromtimestamp(privileges[p])
                except:
                    queryVal = None
            elif p.lower() == 'remove':
                queryVal = None
                queryType = 'remove'
            else:
                raise Exception(f'"{p}" is not a recognized privilege type.')

            if queryType == 'update':
                queryStr = f'''
                    UPDATE aide_admin.authentication
                    SET {p} = %s
                    WHERE username IN %s
                    AND project = %s
                    RETURNING username;
                '''
                result = self.dbConnector.execute(queryStr, (queryVal, tuple(userList), project), 'all')
            else:
                queryStr = '''
                    DELETE FROM aide_admin.authentication
                    WHERE username IN %s
                    AND project = %s
                    RETURNING username;
                '''
                result = self.dbConnector.execute(queryStr, (tuple(userList), project), 'all')
            
            if result is None or not len(result):
                #TODO: provide more sophisticated error response
                return {
                    'status': 2,
                    'message': f'An error occurred while trying to set permission type "{p}"'
                }

        return {
            'status': 0
        }


    def getProjectUsers(self, project):
        '''
            Returns a list of users that are enrolled in the project,
            as well as their roles within the project.
        '''

        queryStr = sql.SQL('SELECT * FROM aide_admin.authentication WHERE project = %s;')
        result = self.dbConnector.execute(queryStr, (project,), 'all')
        response = []
        for r in result:
            user = {}
            for key in r.keys():
                if isinstance(r[key], datetime):
                    val = r[key].timestamp()
                else:
                    val = r[key]
                user[key] = val
            response.append(user)
        return response


    def createProject(self, username, properties):
        '''
            Receives the most basic, mostly non-changeable settings for a new project
            ("properties") with the following entries:
            - shortname
            - owner (the current username)
            - name
            - description
            - annotationType
            - predictionType

            More advanced settings (UI config, AI model, etc.) will be configured after
            the initial project creation stage.

            Verifies whether these settings are available for a new project. If they are,
            it creates a new database schema for the project, adds an entry for it to the
            admin schema table and makes the current user admin. Returns True in this case.
            Otherwise raises an exception.
        '''

        shortname = properties['shortname']

        # verify availability of the project name and shortname
        if not self.getProjectNameAvailable(properties['name']):
            raise Exception('Project name "{}" unavailable.'.format(properties['name']))
        if not self.getProjectShortNameAvailable(shortname):
            raise Exception('Project shortname "{}" unavailable.'.format(shortname))

        # load base SQL
        with open('modules/ProjectAdministration/static/sql/create_schema.sql', 'r') as f:
            queryStr = sql.SQL(f.read())

        
        # determine annotation and prediction types and add fields accordingly
        annotationFields = list(getattr(Fields_annotation, properties['annotationType']).value)
        predictionFields = list(getattr(Fields_prediction, properties['predictionType']).value)

        # custom band configuration
        bandConfig = properties.get('band_config', None)
        if not (isinstance(bandConfig, list) or isinstance(bandConfig, tuple)):
            bandConfig = helpers.DEFAULT_BAND_CONFIG
        bandConfig = json.dumps(bandConfig)

        # custom render configuration
        renderConfig = properties.get('render_config', None)
        if isinstance(renderConfig, dict):
            # verify entries
            renderConfig = helpers.check_args(renderConfig, helpers.DEFAULT_RENDER_CONFIG)
        else:
            # set to default
            renderConfig = helpers.DEFAULT_RENDER_CONFIG
        renderConfig = json.dumps(renderConfig)

        # create project schema
        self.dbConnector.execute(queryStr.format(
                id_schema=sql.Identifier(shortname),
                id_auth=sql.Identifier(self.config.getProperty('Database', 'user')),
                id_image=sql.Identifier(shortname, 'image'),
                id_iu=sql.Identifier(shortname, 'image_user'),
                id_bookmark=sql.Identifier(shortname, 'bookmark'),
                id_labelclassGroup=sql.Identifier(shortname, 'labelclassgroup'),
                id_labelclass=sql.Identifier(shortname, 'labelclass'),
                id_annotation=sql.Identifier(shortname, 'annotation'),
                id_cnnstate=sql.Identifier(shortname, 'cnnstate'),
                id_modellc=sql.Identifier(shortname, 'model_labelclass'),
                id_prediction=sql.Identifier(shortname, 'prediction'),
                id_workflow=sql.Identifier(shortname, 'workflow'),
                id_workflowHistory=sql.Identifier(shortname, 'workflowhistory'),
                id_filehierarchy=sql.Identifier(shortname, 'filehierarchy'),
                id_taskHistory=sql.Identifier(shortname, 'taskhistory'),
                annotation_fields=sql.SQL(', ').join([sql.SQL(field) for field in annotationFields]),
                prediction_fields=sql.SQL(', ').join([sql.SQL(field) for field in predictionFields])
            ),
            None,
            None
        )

        # check if schema got created
        valid = self.dbConnector.execute('''
            SELECT COUNT(*) AS present
            FROM "information_schema".schemata
            WHERE schema_name = %s;
        ''', (shortname,), 1)
        if valid is None or not len(valid) or valid[0]['present'] < 1:
            raise Exception(f'Project with shortname "{shortname}" could not be created.\nCheck for database permission errors.')

        # register project
        self.dbConnector.execute('''
            INSERT INTO aide_admin.project (shortname, name, description,
                owner,
                secret_token,
                interface_enabled,
                annotationType, predictionType,
                isPublic, demoMode,
                ai_alcriterion_library,
                numImages_autotrain,
                minNumAnnoPerImage,
                maxNumImages_train,
                ui_settings,
                band_config,
                render_config)
            VALUES (
                %s, %s, %s,
                %s,
                %s,
                %s,
                %s, %s,
                %s, %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );
            ''',
            (
                shortname,
                properties['name'],
                (properties['description'] if 'description' in properties else ''),
                username,
                secrets.token_urlsafe(32),
                False,
                properties['annotationType'],
                properties['predictionType'],
                False, False,
                'ai.al.builtins.maxconfidence.MaxConfidence',   #TODO: default AL criterion to facilitate auto-training
                128, 0, 128,            #TODO: default values for automated AI model training
                json.dumps(self.defaultUIsettings),
                bandConfig,
                renderConfig
            ),
            None)

        # register user in project
        self.dbConnector.execute('''
                INSERT INTO aide_admin.authentication (username, project, isAdmin)
                VALUES (%s, %s, true);
            ''',
            (username, shortname,),
            None)

        # notify FileServer instance(s) to set up project folders
        process = fileServer_interface.aide_internal_notify.si({
            'task': 'create_project_folders',
            'projectName': shortname
        })
        process.apply_async(queue='aide_broadcast',
                            ignore_result=True)
        
        return True



    def updateProjectSettings(self, project, projectSettings):
        '''
            TODO
        '''

        # check UI settings first
        if 'ui_settings' in projectSettings:
            if isinstance(projectSettings['ui_settings'], str):
                projectSettings['ui_settings'] = json.loads(projectSettings['ui_settings'])
            fieldNames = [
                ('welcomeMessage', str),
                ('numImagesPerBatch', int),
                ('minImageWidth', int),
                ('numImageColumns_max', int),
                ('defaultImage_w', int),
                ('defaultImage_h', int),
                ('styles', dict),
                ('enableEmptyClass', bool),
                ('showPredictions', bool),
                ('showPredictions_minConf', float),
                ('carryOverPredictions', bool),
                ('carryOverRule', str),
                ('carryOverPredictions_minConf', float),
                ('defaultBoxSize_w', int),
                ('defaultBoxSize_h', int),
                ('minBoxSize_w', int),
                ('minBoxSize_h', int),
                ('showImageNames', bool),
                ('showImageURIs', bool)
            ]
            uiSettings_new, uiSettingsKeys_new = helpers.parse_parameters(projectSettings['ui_settings'], fieldNames, absent_ok=True, escape=True)   #TODO: escape
            
            # adopt current settings and replace values accordingly
            uiSettings = self.dbConnector.execute('''SELECT ui_settings
                    FROM aide_admin.project
                    WHERE shortname = %s;            
                ''', (project,), 1)
            uiSettings = json.loads(uiSettings[0]['ui_settings'])
            for kIdx in range(len(uiSettingsKeys_new)):
                if uiSettingsKeys_new[kIdx] not in uiSettings:  #TODO: may be a bit careless, as any new keywords could be added...
                    uiSettings[uiSettingsKeys_new[kIdx]] = uiSettings_new[kIdx]
                elif isinstance(uiSettings[uiSettingsKeys_new[kIdx]], dict):
                    ProjectConfigMiddleware._recursive_update(uiSettings[uiSettingsKeys_new[kIdx]], uiSettings_new[kIdx])
                else:
                    uiSettings[uiSettingsKeys_new[kIdx]] = uiSettings_new[kIdx]

            # auto-complete with defaults where missing
            uiSettings = helpers.check_args(uiSettings, self.defaultUIsettings)

            projectSettings['ui_settings'] = json.dumps(uiSettings)


        # parse remaining parameters
        fieldNames = [
            ('description', str),
            ('isPublic', bool),
            ('secret_token', str),
            ('demoMode', bool),
            ('ui_settings', str),
            ('interface_enabled', bool),
            ('watch_folder_enabled', bool),
            ('watch_folder_remove_missing_enabled', bool)
        ]

        vals, params = helpers.parse_parameters(projectSettings, fieldNames, absent_ok=True, escape=False)
        vals.append(project)

        # commit to DB
        queryStr = sql.SQL('''UPDATE aide_admin.project
            SET
            {}
            WHERE shortname = %s;
            '''
        ).format(
            sql.SQL(',').join([sql.SQL('{} = %s'.format(item)) for item in params])
        )

        self.dbConnector.execute(queryStr, tuple(vals), None)

        return True

    

    def updateClassDefinitions(self, project, classdef, removeMissing=False):
        '''
            Updates the project's class definitions.
            if "removeMissing" is set to True, label classes that are present
            in the database, but not in "classdef," will be removed. Label
            class groups will only be removed if they do not reference any
            label class present in "classdef." This functionality is disallowed
            in the case of segmentation masks.
        '''

        warnings = []

        # check if project contains segmentation masks
        metaType = self.dbConnector.execute('''
                SELECT annotationType, predictionType FROM aide_admin.project
                WHERE shortname = %s;
            ''',
            (project,),
            1
        )[0]
        is_segmentation = any(['segmentationmasks' in m.lower() for m in metaType.values()])
        if is_segmentation:
            # segmentation: we disallow deletion and serial idx > 255
            if removeMissing:
                warnings.append('Pixel-wise segmentation projects disallow removing label classes.')
            removeMissing = False
            lcQuery = self.dbConnector.execute(sql.SQL('''
                SELECT id, idx, color FROM {};
            ''').format(sql.Identifier(project, 'labelclass')), None, 'all')
            colors = dict([[c['color'].lower(), c['id']] for c in lcQuery])
            colors.update({         # we disallow fully black or white colors for segmentation, too
                '#000000': -1,
                '#ffffff': -1
            })

            maxIdx = 0 if not len(lcQuery) else max([l['idx'] for l in lcQuery])
        else:
            colors = {}
            maxIdx = 0

        # get current classes from database
        db_classes = {}
        db_groups = {}
        queryStr = sql.SQL('''
            SELECT * FROM {id_lc} AS lc
            FULL OUTER JOIN (
                SELECT id AS lcgid, name AS lcgname, parent, color
                FROM {id_lcg}
            ) AS lcg
            ON lc.labelclassgroup = lcg.lcgid
        ''').format(
            id_lc=sql.Identifier(project, 'labelclass'),
            id_lcg=sql.Identifier(project, 'labelclassgroup')
        )
        result = self.dbConnector.execute(queryStr, None, 'all')
        for r in result:
            if r['id'] is not None:
                db_classes[r['id']] = r
            if r['lcgid'] is not None:
                if not r['lcgid'] in db_groups:
                    db_groups[r['lcgid']] = {**r, **{'num_children':0}}
                elif not 'lcgid' in db_groups[r['lcgid']]:
                    db_groups[r['lcgid']] = {**db_groups[r['lcgid']], **r}
            if r['labelclassgroup'] is not None:
                if not r['labelclassgroup'] in db_groups:
                    db_groups[r['labelclassgroup']] = {'num_children':1}
                else:
                    db_groups[r['labelclassgroup']]['num_children'] += 1

        # parse provided class definitions list
        unique_keystrokes = set()
        classes_new = []
        classes_update = []
        classgroups_update = []
        def _parse_item(item, parent=None):
            # get or create ID for item
            try:
                itemID = uuid.UUID(item['id'])
            except:
                itemID = uuid.uuid1()
                while itemID in classes_update or itemID in classgroups_update:
                    itemID = uuid.uuid1()

            color = item.get('color', None)
            
            # resolve potentially duplicate/too similar color values
            if isinstance(color, str) and (color not in colors or colors[color] != itemID):
                color = helpers.offsetColor(color.lower(), colors.keys(), self.MINIMAL_COLOR_OFFSET)
            elif not isinstance(color, str):
                color = helpers.randomHexColor(colors.keys(), self.MINIMAL_COLOR_OFFSET)

            color = color.lower()

            entry = {
                'id': itemID,
                'name': item['name'],
                'color': color,
                'keystroke': None,
                'labelclassgroup': parent
            }
            colors[color] = itemID
            if 'children' in item:
                # label class group
                classgroups_update.append(entry)
                for child in item['children']:
                    _parse_item(child, itemID)
            else:
                # label class
                if 'keystroke' in item and not item['keystroke'] in unique_keystrokes:
                    entry['keystroke'] = item['keystroke']
                    unique_keystrokes.add(item['keystroke'])
                if entry.get('id', None) in db_classes:
                    classes_update.append(entry)
                else:
                    classes_new.append(entry)

        for item in classdef:
            _parse_item(item, None)
        
        # apply changes
        if removeMissing:
            queryArgs = []
            if len(classes_update):
                # remove all missing label classes
                lcSpec = sql.SQL('WHERE id NOT IN %s')
                queryArgs.append(tuple([(l['id'],) for l in classes_update]))
            else:
                # remove all label classes
                lcgSpec = sql.SQL('')
            if len(classgroups_update):
                # remove all missing labelclass groups
                lcgSpec = sql.SQL('WHERE id NOT IN %s')
                queryArgs.append(tuple([(l['id'],) for l in classgroups_update]))
            else:
                # remove all labelclass groups
                lcgSpec = sql.SQL('')
            queryStr = sql.SQL('''
                DELETE FROM {id_lc}
                {lcSpec};
                DELETE FROM {id_lcg}
                {lcgSpec};
            ''').format(
                id_lc=sql.Identifier(project, 'labelclass'),
                id_lcg=sql.Identifier(project, 'labelclassgroup'),
                lcSpec=lcSpec,
                lcgSpec=lcgSpec
            )
            self.dbConnector.execute(queryStr, tuple(queryArgs), None)
        
        # add/update in order (groups, set their parents, label classes)
        groups_new = [(g['id'], g['name'], g['color'],) for g in classgroups_update]
        queryStr = sql.SQL('''
            INSERT INTO {id_lcg} (id, name, color)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                color = EXCLUDED.color;
        ''').format(        #TODO: on conflict(name)
            id_lcg=sql.Identifier(project, 'labelclassgroup')
        )
        self.dbConnector.insert(queryStr, groups_new)

        # set parents
        groups_parents = [(g['id'], g['labelclassgroup'],) for g in classgroups_update if ('labelclassgroup' in g and g['labelclassgroup'] is not None)]
        queryStr = sql.SQL('''
            UPDATE {id_lcg} AS lcg
            SET parent = q.parent
            FROM (VALUES %s) AS q(id, parent)
            WHERE lcg.id = q.id;
        ''').format(
            id_lcg=sql.Identifier(project, 'labelclassgroup')
        )
        self.dbConnector.insert(queryStr, groups_parents)

        # update existing label classes
        if is_segmentation and maxIdx >= 255 and len(classes_new):
            # segmentation and maximum labelclass idx serial reached: cannot add new classes
            warnings.append('Maximum class index ordinal 255 reached. The following new classes had to be discarded: {}.'.format(
                ','.join(['"{}"'.format(c['name']) for c in classes_new])
            ))
            classes_new = []
        else:
            # do updates and insertions in one go
            classes_update.extend(classes_new)

        lcdata = [(l['id'], l['name'], l['color'], l['keystroke'], l['labelclassgroup'],) for l in classes_update]
        queryStr = sql.SQL('''
            INSERT INTO {id_lc} (id, name, color, keystroke, labelclassgroup)
            VALUES %s
            ON CONFLICT (id) DO UPDATE
            SET name = EXCLUDED.name,
            color = EXCLUDED.color,
            keystroke = EXCLUDED.keystroke,
            labelclassgroup = EXCLUDED.labelclassgroup;
        ''').format(    #TODO: on conflict(name)
            id_lc=sql.Identifier(project, 'labelclass')
        )
        self.dbConnector.insert(queryStr, lcdata)

        return warnings

    

    def getModelToProjectClassMapping(self, project, aiModelID=None):
        '''
            Returns a dict of tuples of tuples (AI model label class name,
            project label class ID), organized by AI model library.
            These label class mappings are used to translate from AI model
            state class predictions from the Model Marketplace to label class
            IDs present in the current project.
            If "aiModelID" is provided (str or Iterable of str), only
            definitions for the provided AI model libraries are returned.
        '''
        if aiModelID is None or not isinstance(aiModelID, Iterable):
            libStr = sql.SQL('')
        else:
            if isinstance(aiModelID, str):
                aiModelID = (uuid.UUID(aiModelID),)
            elif isinstance(aiModelID, uuid.UUID):
                aiModelID = (aiModelID,)
            elif isinstance(aiModelID, Iterable):
                aiModelID = list(aiModelID)
                for aIdx in range(len(aiModelID)):
                    if not isinstance(aiModelID[aIdx], uuid.UUID):
                        aiModelID[aIdx] = uuid.UUID(aiModelID)
            libStr = sql.SQL('WHERE marketplace_origin_id IN (%s)')
        
        response = {}
        result = self.dbConnector.execute(sql.SQL(
            'SELECT * FROM {id_modellc} {libStr};'
        ).format(
            id_modellc=sql.Identifier(project, 'model_labelclass'),
            libStr=libStr
        ), aiModelID, 'all')
        if result is not None and len(result):
            for r in result:
                modelID = str(r['marketplace_origin_id'])
                if modelID not in response:
                    response[modelID] = []
                lcProj = (str(r['labelclass_id_project']) if r['labelclass_id_project'] is not None else None)
                response[modelID].append((r['labelclass_id_model'], r['labelclass_name_model'], lcProj))
        return response


    
    def saveModelToProjectClassMapping(self, project, mapping):
        '''
            Receives a dict of tuples of tuples, organized by AI model library, and saves
            the information in the database.
            NOTE: all previous rows in the database for the given AI model
            library entries are deleted prior to insertion of the new values.
        '''
        # assemble arguments
        aiModelIDs = set()
        values = []
        labelclasses_new = {}       # label classes in model to add new to project
        for aiModelID in mapping.keys():
            aiModelIDs.add(uuid.UUID(aiModelID))
            nextMap = mapping[aiModelID]
            for row in nextMap:
                # tuple order in map: (source class ID, source class name, target class ID)
                sourceID = row[0]
                sourceName = row[1]
                targetID = None
                if isinstance(row[2], str):
                    if row[2].lower() == '$$add_new$$':
                        # special flag to add new labelclass to project
                        labelclasses_new[sourceName] = (uuid.UUID(aiModelID), sourceID, sourceName, targetID)
                        continue    # we'll deal with newly added classes later
                    else:
                        try:
                            targetID = uuid.UUID(row[2])
                        except:
                            targetID = None
                values.append((uuid.UUID(aiModelID), sourceID, sourceName, targetID))
        
        # add any newly added label classes to project
        if len(labelclasses_new):
            lc_added = self.dbConnector.insert(sql.SQL('''
                INSERT INTO {id_lc} (name, color)
                VALUES %s
                RETURNING id, name;
            ''').format(id_lc=sql.Identifier(project, 'labelclass')),
            [(l[2],helpers.randomHexColor(),) for l in labelclasses_new.values()], 'all')       #TODO: make random colors exclusive from each other
            for row in lc_added:
                values.append((uuid.UUID(aiModelID), labelclasses_new[row[1]][1], labelclasses_new[row[1]][2], row[0]))

        # perform insertion
        self.dbConnector.insert(sql.SQL('''
            DELETE FROM {id_modellc} WHERE
            marketplace_origin_id IN %s;
        ''').format(
            id_modellc=sql.Identifier(project, 'model_labelclass')
        ),
        (tuple(aiModelIDs),))
        self.dbConnector.insert(sql.SQL('''
            INSERT INTO {id_modellc} (marketplace_origin_id, labelclass_id_model, labelclass_name_model,
            labelclass_id_project)
            VALUES %s;
        ''').format(
            id_modellc=sql.Identifier(project, 'model_labelclass')
        ),
        tuple(values))
        return 0



    def getProjectNameAvailable(self, projectName):
        '''
            Returns True if the provided project (long) name is available.
        '''
        if not isinstance(projectName, str):
            return False
        projectName_stripped = projectName.strip().lower()
        if not len(projectName_stripped):
            return False

        # check if name matches prohibited AIDE keywords (we do not replace long names)
        if projectName_stripped in self.PROHIBITED_STRICT or any([p in projectName_stripped for p in self.PROHIBITED_STRICT]):
            return False
        if projectName_stripped in self.PROHIBITED_NAMES:
            return False
        if any([projectName_stripped.startswith(p) for p in self.PROHIBITED_NAME_PREFIXES]):
            return False

        # check if name is already taken
        result = self.dbConnector.execute('''SELECT 1 AS result
            FROM aide_admin.project
            WHERE name = %s;
            ''',
            (projectName,),
            1)
        
        if result is None or not len(result):
            return True
        else:
            return result[0]['result'] != 1


    def getProjectShortNameAvailable(self, projectName):
        '''
            Returns True if the provided project shortname is available.
            In essence, "available" means that a database schema with the given
            name can be created (this includes Postgres schema name conventions).
            Returns False otherwise.
        '''
        if not isinstance(projectName, str):
            return False
        projectName_stripped = projectName.strip().lower()
        if not len(projectName_stripped):
            return False

        # check if name matches prohibited AIDE keywords; replace where possible
        if projectName_stripped in self.PROHIBITED_STRICT or any([p in projectName_stripped for p in self.PROHIBITED_STRICT]):
            return False
        if projectName_stripped in self.PROHIBITED_NAMES or projectName_stripped in self.PROHIBITED_SHORTNAMES:
            return False
        if any([projectName_stripped.startswith(p) for p in self.PROHIBITED_NAME_PREFIXES]):
            return False
        for p in self.SHORTNAME_PATTERNS_REPLACE:
            projectName = projectName.replace(p, '_')

        # check if provided name is valid as per Postgres conventions
        matches = re.findall('(^(pg_|[0-9]).*|.*(\$|\s)+.*)', projectName)
        if len(matches):
            return False

        # check if project shorthand already exists in database
        result = self.dbConnector.execute('''SELECT 1 AS result
            FROM information_schema.schemata
            WHERE schema_name ilike %s
            UNION ALL
            SELECT 1 FROM aide_admin.project
            WHERE shortname ilike %s;
            ''',
            (projectName,projectName,),
            2)

        if result is None or not len(result):
            return True

        if len(result) == 2:
            return result[0]['result'] != 1 and result[1]['result'] != 1
        elif len(result) == 1:
            return result[0]['result'] != 1
        else:
            return True


    def getProjectArchived(self, project, username):
        '''
            Returns the "archived" flag of a project.
            Throws an error if user is not registered in project,
            or if the project is not in demo mode.
        '''

        # check if user is authenticated
        isAuthenticated = self.dbConnector.execute('''
            SELECT username, isSuperUser FROM (
                SELECT username
                FROM aide_admin.authentication
                WHERE project = %s
            ) AS auth
            RIGHT OUTER JOIN (
                SELECT name, isSuperUser
                FROM aide_admin.user
                WHERE name = %s
            ) AS usr
            ON auth.username = usr.name
        ''', (project, username), 1)
        
        if isAuthenticated is None or not len(isAuthenticated) or \
            (isAuthenticated[0]['username'] is None and not isAuthenticated[0]['issuperuser']):
            # project does not exist or user is neither member nor super user
            return {
                'status': 2,
                'message': 'User cannot view project details.'
            }
        
        isArchived = self.dbConnector.execute('''
            SELECT archived FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)

        if isArchived is None or not len(isArchived):
            return {
                'status': 3,
                'message': 'Project does not exist.'
            }

        return {
            'status': 0,
            'archived': isArchived[0]['archived']
        }
        

    def setProjectArchived(self, project, username, archived):
        '''
            Archives or unarchives a project by setting the "archived" flag in the database
            to the value in "archived".
            An archived project is simply hidden from the list and unchangeable, but stays
            intact as-is and can be unarchived if needed. No data is deleted.

            Only project owners and super users can archive projects (i.e., even being a
            project administrator is not enough).
        '''

        # check if user is authenticated
        isAuthenticated = self.dbConnector.execute('''
            SELECT CASE WHEN owner = %s THEN TRUE ELSE FALSE END AS result
            FROM aide_admin.project
            WHERE shortname = %s
            UNION ALL
            SELECT isSuperUser AS result
            FROM aide_admin.user
            WHERE name = %s;
        ''', (username, project, username), 2)
        
        if not isAuthenticated[0]['result'] \
            and not isAuthenticated[1]['result']:
            # user is neither project owner nor super user; reject
            return {
                'status': 2,
                'message': 'User is not authenticated to archive or unarchive project.'
            }

        # archive project
        self.dbConnector.execute('''
            UPDATE aide_admin.project
            SET ARCHIVED = %s
            WHERE shortname = %s;
        ''', (archived, project), None)

        return {
            'status': 0
        }


    def deleteProject(self, project, username, deleteFiles=False):
        '''
            Removes a project from the database, including all metadata.
            Also dispatches a Celery task to the FileServer to
            delete images (and other project-specific data on disk)
            if "deleteFiles" is True.
            WARNING: this seriously deletes a project in its entirety; any data will be
            lost forever.

            Only project owners and super users can delete projects (i.e., even being a
            project administrator is not enough).
        '''

        # check if user is authenticated
        isAuthenticated = self.dbConnector.execute('''
            SELECT CASE WHEN owner = %s THEN TRUE ELSE FALSE END AS result
            FROM aide_admin.project
            WHERE shortname = %s
            UNION ALL
            SELECT isSuperUser AS result
            FROM aide_admin.user
            WHERE name = %s;
        ''', (username, project, username), 2)
        
        if not isAuthenticated[0]['result'] \
            and not isAuthenticated[1]['result']:
            # user is neither project owner nor super user; reject
            return {
                'status': 2,
                'message': 'User is not authenticated to delete project.'
            }

        # check if project exists; if not it may already be deleted
        projectExists = self.dbConnector.execute('''
            SELECT shortname
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project,), 1)
        if projectExists is None or not len(projectExists):
            # project does not exist; return        #TODO: still allow deleting files on disk?
            return {
                'status': 3,
                'message': 'Project does not exist.'
            }

        # stop ongoing tasks
        #TODO: make more elegant
        tc = TaskCoordinatorMiddleware(self.config, self.dbConnector)
        tc.revokeAllJobs(project, username, includeAItasks=True)


        # remove rows from database
        self.dbConnector.execute('''
            DELETE FROM aide_admin.authentication
            WHERE project = %s;
            DELETE FROM aide_admin.project
            WHERE shortname = %s;
        ''', (project, project,), None)
        
        # dispatch Celery task to remove DB schema and files (if requested)
        process = fileServer_interface.deleteProject.si(project, deleteFiles)
        process.apply_async(queue='FileServer')       #TODO: task ID? Progress monitoring?

        #TODO: return Celery task ID?
        return {
                'status': 0
            }