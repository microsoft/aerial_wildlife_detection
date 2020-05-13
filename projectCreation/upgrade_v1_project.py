'''
    Helper script to upgrade an existing project from AIDE v1 to a v2 (multi-project)
    installation.

    Use as follows:
        1. Move the images from your v1 project to a sub-folder of v2's "file directory"
           with the name of your v1 project's database schema (check parameter "staticfiles_dir"
           under "[FileServer]" of the v2's configuration .ini file).
           For example:
                If your v1's images are stored under "/datadrive/images",
                your v1's database schema is named "my_great_project",
                and the "staticfiles_dir" of v2 is set to "/datadrive/new",
                you have to move all images from "/datadrive/images" to
                "/datadrive/new/my_great_project".
           Alternatively, the script may ask you if it should create a symbolic link from the new
           target place to the original folder. This is a temporary solution only. Also, you should
           avoid using recursive links (e.g., "/datadrive/images/my_great_project" pointing to
           "/datadrive/images").
        1. Make sure the "AIDE_CONFIG_PATH" environment variable points to the
           configuration .ini file of the NEW (v2) installation.
        2. Run this script as follows:
            python projectCreation/upgrade_v1_project.py --settings_filepath=/path/to/settings.ini
           
           replace "/path/to/settings.ini" with the file path to the configuration .ini file
           of your original (v1) installation.

    NOTES:
        - This script does NOT transfer a v1 project to another database, it just upgrades an
          existing v1 project to work with the v2 installation that points to the SAME database.
        - Once a v1 project has been successfully upgraded with this script, the original
          configuration .ini file can be discarded. There is no need to run this script more than
          one time for a specific v1 project.
        - THERE IS NO GUARANTEE THAT A v1 PROJECT, CONVERTED TO v2 WITH THIS SCRIPT, WILL STILL
          RUN UNDER A v1 INSTALLATION. IT IS STRONGLY DISCOURAGED TO CONTINUE TO USE THE v1 SOFT-
          WARE ON A PROJECT THAT HAS BEEN MIGRATED.

    
    2019-20 Benjamin Kellenberger
'''

import os
import argparse
import json
import secrets
from setup.migrate_aide import migrate_aide, MODIFICATIONS_sql
from urllib.parse import urlparse
from modules import UserHandling



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Upgrade and register an AIDE v1 project to an existing v2 installation.')
    parser.add_argument('--settings_filepath', type=str,
                    help='Path of the configuration .ini file for the v1 project to be upgraded to v2.')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        raise Exception('ERROR: System environment variable "AIDE_CONFIG_PATH" must be set and must point to the configuration .ini file of the v2 installation.')
    if not 'AIDE_MODULES' in os.environ:
        os.environ['AIDE_MODULES'] = ''     # for compatibility with Celery worker import

    from util.configDef import Config
    from modules import Database

    config = Config()
    dbConn = Database(config)
    if dbConn.connectionPool is None:
        raise Exception('Error connecting to database.')

    # db schema of v1 project
    dbSchema = config.getProperty('Database', 'schema')

    # update tables: make modifications one at a time
    for mod in MODIFICATIONS_sql:
        dbConn.execute(mod.format(schema=dbSchema), None, None)


    # migrate to v2

    # assemble dict of dynamic project UI and AI settings
    try:
        with open(config.getProperty('LabelUI', 'styles_file', type=str, fallback='modules/LabelUI/static/json/styles.json'), 'r') as f:
            styles = json.load(f)
            styles = styles['styles']
    except:
        styles = {}
    try:
        with open(config.getProperty('Project', 'backdrops_file', type=str, fallback='modules/LabelUI/static/json/backdrops.json'), 'r') as f:
            backdrops = json.load(f)
    except:
        backdrops = {}
    try:
        with open(config.getProperty('Project', 'welcome_message_file', type=str, fallback='modules/LabelUI/static/templates/welcome_message.html'), 'r') as f:
            welcomeMessage = f.readlines()
    except:
        welcomeMessage = ''
    uiSettings = {
        'enableEmptyClass': config.getProperty('Project', 'enableEmptyClass', fallback='no'),
        'showPredictions': config.getProperty('LabelUI', 'showPredictions', fallback='yes'),
        'showPredictions_minConf': config.getProperty('LabelUI', 'showPredictions_minConf', type=float, fallback=0.5),
        'carryOverPredictions': config.getProperty('LabelUI', 'carryOverPredictions', fallback='no'),
        'carryOverRule': config.getProperty('LabelUI', 'carryOverRule', fallback='maxConfidence'),
        'carryOverPredictions_minConf': config.getProperty('LabelUI', 'carryOverPredictions_minConf', type=float, fallback=0.75),
        'defaultBoxSize_w': config.getProperty('LabelUI', 'defaultBoxSize_w', type=int, fallback=10),
        'defaultBoxSize_h': config.getProperty('LabelUI', 'defaultBoxSize_h', type=int, fallback=10),
        'minBoxSize_w': config.getProperty('Project', 'box_minWidth', type=int, fallback=1),
        'minBoxSize_h': config.getProperty('Project', 'box_minHeight', type=int, fallback=1),
        'numImagesPerBatch': config.getProperty('LabelUI', 'numImagesPerBatch', type=int, fallback=1),
        'minImageWidth': config.getProperty('LabelUI', 'minImageWidth', type=int, fallback=300),
        'numImageColumns_max': config.getProperty('LabelUI', 'numImageColumns_max', type=int, fallback=1),
        'defaultImage_w': config.getProperty('LabelUI', 'defaultImage_w', type=int, fallback=800),
        'defaultImage_h': config.getProperty('LabelUI', 'defaultImage_h', type=int, fallback=600),
        'styles': styles,
        'backdrops': backdrops,
        'welcomeMessage': welcomeMessage
    }

    # models
    modelPath = config.getProperty('AIController', 'model_lib_path', fallback=None)
    if modelPath is not None and len(modelPath):
        dbConn.execute('UPDATE {schema}.cnnstate SET model_library = %s WHERE model_library IS NULL;'.format(schema=dbSchema),
        (modelPath,), None)
    else:
        modelPath = None
    alCriterionPath = config.getProperty('AIController', 'al_criterion_lib_path', fallback=None)
    if alCriterionPath is None or not len(alCriterionPath): alCriterionPath = None

    modelSettingsPath = config.getProperty('AIController', 'model_options_path', fallback=None)
    if modelSettingsPath is not None and len(modelSettingsPath):
        try:
            with open(modelSettingsPath, 'r') as f:
                modelSettings = json.load(f).dumps()
        except:
            print('WARNING: could not parse settings defined in model settings path ("{}")'.format(modelSettingsPath))
            modelSettings = None
    else:
        modelSettings = None
    alCriterionSettingsPath = config.getProperty('AIController', 'al_criterion_options_path', fallback=None)
    if alCriterionSettingsPath is not None and len(alCriterionSettingsPath):
        try:
            with open(alCriterionSettingsPath, 'r') as f:
                alCriterionSettings = json.load(f).dumps()
        except:
            print('WARNING: could not parse settings defined in AL criterion settings path ("{}")'.format(alCriterionSettingsPath))
            alCriterionSettings = None
    else:
        alCriterionSettings = None


    #TODO: eventually replace legacy fields with default workflow
    # # workflows: move values from legacy fields and create new workflow, if project already registered
    # colnames = dbConn.execute('''
    #     SELECT column_name
    #     FROM information_schema.columns
    #     WHERE table_schema = 'aide_admin'
    #     AND table_name   = 'project';
    # ''', None, 'all')
    # colnames = set([c['column_name'] for c in colnames])
    # if 'minnumannoperimage' in colnames:
    #     # legacy fields present; read and replace with new default workflow
    #     autoTrainSpec = dbConn.execute('''
    #         SELECT minnumannoperimage, maxnumimages_train, maxnumimages_inference
    #         FROM aide_admin.project
    #         WHERE shortname = %s;
    #     ''', (dbSchema,), 1)
    #     if len(autoTrainSpec):
    #         autoTrainSpec = autoTrainSpec[0]
    #         #TODO


    # register project
    secretToken = secrets.token_urlsafe(32)
    dbConn.execute('''
        INSERT INTO aide_admin.project (shortname, name, description,
            owner,
            secret_token,
            interface_enabled,
            demoMode,
            annotationType, predictionType, ui_settings,
            numImages_autoTrain,
            minNumAnnoPerImage,
            maxNumImages_train,
            maxNumImages_inference,
            ai_model_enabled,
            ai_model_library, ai_model_settings,
            ai_alCriterion_library, ai_alCriterion_settings
            )
        VALUES (
            %s, %s, %s,
            %s,
            %s,
            %s,
            %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (shortname) DO NOTHING;
    ''',
        (
            dbSchema,
            config.getProperty('Project', 'projectName'),
            config.getProperty('Project', 'projectDescription'),
            config.getProperty('Project', 'adminName'),
            secretToken,
            True,
            config.getProperty('Project', 'demoMode'),
            config.getProperty('Project', 'annotationType'),
            config.getProperty('Project', 'predictionType'),
            json.dumps(uiSettings),
            config.getProperty('AIController', 'numImages_autoTrain'),
            config.getProperty('AIController', 'minNumAnnoPerImage'),
            config.getProperty('AIController', 'maxNumImages_train'),
            config.getProperty('AIController', 'maxNumImages_inference'),
            (modelPath is not None),
            modelPath, modelSettings,
            alCriterionPath, alCriterionSettings
        )
    )
    dbConn.execute('''DO
        $do$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM   information_schema.tables 
                WHERE  table_schema = '{schema}'
                AND    table_name = 'user'
            ) THEN
                INSERT INTO aide_admin.authentication (username, project, isAdmin)
                    SELECT name, %s AS project, isAdmin FROM {schema}.user;
            END IF;
        END $do$;
        '''.format(schema=dbSchema),
        (dbSchema,), None)


    # The multi-project AIDE setup requires images to be in a subfolder named after
    # the project shorthand. Here we tell the user about moving the files, or else
    # propose a temporary fix (softlink).
    softlinkName = config.getProperty('FileServer', 'staticfiles_dir')
    if not os.path.isdir(softlinkName):
        # not running on file server; show message
        print('You do not appear to be running AIDE on a "FileServer" instance.')
        print('INFO: In the process of AIDE supporting multiple projects, each')
        print('project\'s files must be put in a sub-folder named after the project\'s')
        print(f'shorthand (i.e.: {softlinkName}/{dbSchema}/...).')
        print('Make sure to move the files to the new path on the FileServer instance.')
    
    else:
        softlinkName = os.path.join(softlinkName, dbSchema)
        if os.path.islink(softlinkName):
            print(f'INFO: Detected link to project file directory ({softlinkName})')
            print('You might want to move the files to a dedicated folder at some point...')
        else:
            print('INFO: In the process of AIDE supporting multiple projects, each')
            print('project\'s files must be put in a sub-folder named after the project\'s')
            print(f'shorthand (i.e.: {softlinkName}/<images>).')
            print('Ideally, you would want to move the images to that folder, but as a')
            print('temporary fix, you can also use a softlink:')
            print('{} -> {}'.format(softlinkName, config.getProperty('FileServer', 'staticfiles_dir')))
            print('Would you like to create this softlink now?')
            confirmation = None
            while confirmation is None:
                try:
                    confirmation = input('[Y/n]: ')
                    if 'Y' in confirmation:
                        confirmation = True
                    elif 'n' in confirmation.lower():
                        confirmation = False
                    else: raise Exception('Invalid value')
                except:
                    confirmation = None
            if confirmation:
                os.symlink(
                    config.getProperty('FileServer', 'staticfiles_dir'),
                    softlinkName
                )

    # add admin user (if not already present)
    adminName = config.getProperty('Project', 'adminName', type=str, fallback=None)
    if adminName is not None:
        adminEmail = config.getProperty('Project', 'adminEmail')
        adminPass = config.getProperty('Project', 'adminPassword')
        uHandler = UserHandling.backend.middleware.UserMiddleware(config)
        adminPass = uHandler._create_hash(adminPass.encode('utf8'))
        dbConn.execute('''
                INSERT INTO aide_admin.user (name, email, hash, issuperuser)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING;
            ''',
            (adminName, adminEmail, adminPass, True),
            None
        )

    # add authentication
    dbConn.execute('''
            DO
            $do$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM   information_schema.tables 
                    WHERE  table_schema = '{schema}'
                    AND    table_name = 'user'
                ) THEN
                    INSERT INTO aide_admin.authentication (username, project, isAdmin)
                    SELECT name, '{schema}', isAdmin FROM {schema}.user
                    WHERE name IN (SELECT name FROM aide_admin.user)
                    ON CONFLICT (username, project) DO NOTHING;
                END IF;
            END $do$;
        '''.format(schema=dbSchema),
        None,
        None)

    print('Project "{}" has been converted to AIDE v2 standards. Please do not use a v1 installation on this project anymore.'.format(config.getProperty('Project', 'projectName')))