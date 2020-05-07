'''
    Run this file whenever you update AIDE to bring your existing project setup up-to-date
    with respect to changes due to newer versions.
    
    2019-20 Benjamin Kellenberger
'''

import os
import argparse
import json
import secrets
from urllib.parse import urlparse


MODIFICATIONS_sql = [
    'ALTER TABLE {schema}.annotation ADD COLUMN IF NOT EXISTS meta VARCHAR; ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS meta VARCHAR;',
    'ALTER TABLE {schema}.labelclass ADD COLUMN IF NOT EXISTS keystroke SMALLINT UNIQUE;',
    'ALTER TABLE {schema}.image ADD COLUMN IF NOT EXISTS last_requested TIMESTAMPTZ;',

    # support for multiple projects
    'CREATE SCHEMA IF NOT EXISTS aide_admin',
    '''DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'labeltype') THEN
                create type labelType AS ENUM ('labels', 'points', 'boundingBoxes', 'segmentationMasks');
            END IF;
        END
        $$;''',
    '''CREATE TABLE IF NOT EXISTS aide_admin.project (
        shortname VARCHAR UNIQUE NOT NULL,
        name VARCHAR UNIQUE NOT NULL,
        description VARCHAR,
        isPublic BOOLEAN DEFAULT FALSE,
        secret_token VARCHAR,
        interface_enabled BOOLEAN DEFAULT FALSE,
        demoMode BOOLEAN DEFAULT FALSE,
        annotationType labelType NOT NULL,
        predictionType labelType,
        ui_settings VARCHAR,
        numImages_autoTrain BIGINT,
        minNumAnnoPerImage INTEGER,
        maxNumImages_train BIGINT,
        maxNumImages_inference BIGINT,
        ai_model_enabled BOOLEAN NOT NULL DEFAULT FALSE,
        ai_model_library VARCHAR,
        ai_model_settings VARCHAR,
        ai_alCriterion_library VARCHAR,
        ai_alCriterion_settings VARCHAR,
        PRIMARY KEY(shortname)
    );''',
    '''CREATE TABLE IF NOT EXISTS aide_admin.user (
        name VARCHAR UNIQUE NOT NULL,
        email VARCHAR,
        hash BYTEA,
        isSuperuser BOOLEAN DEFAULT FALSE,
        canCreateProjects BOOLEAN DEFAULT FALSE,
        session_token VARCHAR,
        last_login TIMESTAMPTZ,
        secret_token VARCHAR DEFAULT md5(random()::text),
        PRIMARY KEY (name)
    );''',
    'ALTER TABLE aide_admin.user ADD COLUMN IF NOT EXISTS secret_token VARCHAR DEFAULT md5(random()::text);',
    '''CREATE TABLE IF NOT EXISTS aide_admin.authentication (
        username VARCHAR NOT NULL,
        project VARCHAR NOT NULL,
        isAdmin BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (username, project),
        FOREIGN KEY (username) REFERENCES aide_admin.user (name),
        FOREIGN KEY (project) REFERENCES aide_admin.project (shortname)
    );''',
    'ALTER TABLE {schema}.image_user DROP CONSTRAINT IF EXISTS image_user_image_fkey;',
    'ALTER TABLE {schema}.image_user DROP CONSTRAINT IF EXISTS image_user_username_fkey;',
    '''DO
        $do$
        BEGIN
        IF EXISTS (
            SELECT 1
            FROM   information_schema.tables 
            WHERE  table_schema = '{schema}'
            AND    table_name = 'user'
        ) THEN
            INSERT INTO aide_admin.user (name, email, hash, isSuperUser, canCreateProjects, secret_token)
            SELECT name, email, hash, false AS isSuperUser, false AS canCreateProjects, md5(random()::text) AS secret_token FROM {schema}.user
            ON CONFLICT(name) DO NOTHING;
        END IF;
    END $do$;''',
    'ALTER TABLE {schema}.image_user ADD CONSTRAINT image_user_image_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE {schema}.annotation DROP CONSTRAINT IF EXISTS annotation_username_fkey;',
    'ALTER TABLE {schema}.annotation ADD CONSTRAINT annotation_username_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE {schema}.cnnstate ADD COLUMN IF NOT EXISTS model_library VARCHAR;',
    'ALTER TABLE {schema}.cnnstate ADD COLUMN IF NOT EXISTS alCriterion_library VARCHAR;',
    'ALTER TABLE {schema}.image ADD COLUMN IF NOT EXISTS isGoldenQuestion BOOLEAN NOT NULL DEFAULT FALSE;',
    '''-- IoU function for statistical evaluations
    CREATE OR REPLACE FUNCTION "intersection_over_union" (
        "ax" real, "ay" real, "awidth" real, "aheight" real,
        "bx" real, "by" real, "bwidth" real, "bheight" real)
    RETURNS real AS $iou$
        DECLARE
            iou real;
        BEGIN
            SELECT (
                CASE WHEN aright < bleft OR bright < aleft OR
                    atop < bbottom OR btop < abottom THEN 0.0
                ELSE GREATEST(inters / (unionplus - inters), 0.0)
                END
            ) INTO iou
            FROM (
                SELECT 
                    ((iright - ileft) * (itop - ibottom)) AS inters,
                    aarea + barea AS unionplus,
                    aleft, aright, atop, abottom,
                    bleft, bright, btop, bbottom
                FROM (
                    SELECT
                        ((aright - aleft) * (atop - abottom)) AS aarea,
                        ((bright - bleft) * (btop - bbottom)) AS barea,
                        GREATEST(aleft, bleft) AS ileft,
                        LEAST(atop, btop) AS itop,
                        LEAST(aright, bright) AS iright,
                        GREATEST(abottom, bbottom) AS ibottom,
                        aleft, aright, atop, abottom,
                        bleft, bright, btop, bbottom
                    FROM (
                        SELECT (ax - awidth/2) AS aleft, (ay + aheight/2) AS atop,
                            (ax + awidth/2) AS aright, (ay - aheight/2) AS abottom,
                            (bx - bwidth/2) AS bleft, (by + bheight/2) AS btop,
                            (bx + bwidth/2) AS bright, (by - bheight/2) AS bbottom
                    ) AS qq
                ) AS qq2
            ) AS qq3;
            RETURN iou;
        END;
    $iou$ LANGUAGE plpgsql;
    ''',
    'ALTER TABLE {schema}.image ADD COLUMN IF NOT EXISTS date_added TIMESTAMPTZ NOT NULL DEFAULT NOW();',
    'ALTER TABLE aide_admin.authentication ADD COLUMN IF NOT EXISTS admitted_until TIMESTAMPTZ;',
    'ALTER TABLE aide_admin.authentication ADD COLUMN IF NOT EXISTS blocked_until TIMESTAMPTZ;',
    '''ALTER TABLE {schema}.labelclassgroup DROP CONSTRAINT IF EXISTS labelclassgroup_name_unique;
    ALTER TABLE {schema}.labelclassgroup ADD CONSTRAINT labelclassgroup_name_unique UNIQUE (name);''',
    'ALTER TABLE {schema}.image ADD COLUMN IF NOT EXISTS corrupt BOOLEAN;',
    '''
    CREATE TABLE IF NOT EXISTS {schema}.workflow (
        id uuid DEFAULT uuid_generate_v4(),
        name VARCHAR UNIQUE,
        workflow VARCHAR NOT NULL,
        username VARCHAR NOT NULL,
        timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        timeModified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (id),
        FOREIGN KEY (username) REFERENCES aide_admin.user(name)
    )
    ''',
    'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS default_workflow uuid',
    'ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS num_interactions INTEGER NOT NULL DEFAULT 0;'
    'ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS first_checked TIMESTAMPTZ;',
    'ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS total_time_required BIGINT;'
]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update AIDE database structure.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)
    if not 'AIDE_MODULES' in os.environ:
        os.environ['AIDE_MODULES'] = ''     # for compatibility with Celery worker import

    from util.configDef import Config
    from modules import Database

    config = Config()
    dbConn = Database(config)
    if dbConn.connectionPool is None:
        raise Exception('Error connecting to database.')
    dbSchema = config.getProperty('Database', 'schema')


    # make modifications one at a time
    for mod in MODIFICATIONS_sql:
        dbConn.execute(mod.format(schema=dbSchema), None, None)


    # for migration to multiple project support

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
        print('shorthand (i.e.: {}/{}/<images>).'.format(
            softlinkName,
            dbSchema))
        print('Make sure to run this script on all "FileServer" instance(s) as well')
        print('to address this issue.')
    
    else:
        softlinkName = os.path.join(softlinkName, dbSchema)
        if os.path.islink(softlinkName):
            print('INFO: Detected link to project file directory ({})'.format(softlinkName))
            print('You might want to move the files to a dedicated folder at some point...')
        else:
            print('INFO: In the process of AIDE supporting multiple projects, each')
            print('project\'s files must be put in a sub-folder named after the project\'s')
            print('shorthand (i.e.: {}/<images>).'.format(softlinkName))
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

    
    # check fileServer URI
    fileServerURI = config.getProperty('FileServer', 'staticfiles_uri')
    if fileServerURI is not None and len(fileServerURI):
        print('WARNING: please update entry "dataServer_uri" under "[Server]" in the configuration.ini file.')
        print('The latest version of AIDE only specifies the base file server address and port, but nothing else.')
        print('Example:\n')
        print('[Server]')
        print('dataServer_uri = http://fileserver.domain.info:8080')
        print('\nIf you do not use a dedicated file server, you can leave "dataServer_uri" blank.')


    print('Project "{}" is now up-to-date for the latest changes in AIDE.'.format(config.getProperty('Project', 'projectName')))