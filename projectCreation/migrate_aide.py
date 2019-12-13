'''
    Run this file whenever you update AIde to bring your existing project setup up-to-date
    with respect to changes due to newer versions.
    
    2019 Benjamin Kellenberger
'''

import os
import argparse
import json
import secrets


MODIFICATIONS_sql = [
    'ALTER TABLE {schema}.annotation ADD COLUMN IF NOT EXISTS meta VARCHAR; ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS meta VARCHAR;',
    'ALTER TABLE {schema}.labelclass ADD COLUMN IF NOT EXISTS keystroke SMALLINT UNIQUE;',

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
        PRIMARY KEY (name)
    );''',
    '''CREATE TABLE IF NOT EXISTS aide_admin.authentication (
        username VARCHAR NOT NULL,
        project VARCHAR NOT NULL,
        isAdmin BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (username, project),
        FOREIGN KEY (username) REFERENCES aide_admin.user (name),
        FOREIGN KEY (project) REFERENCES aide_admin.project (shortname)
    );''',
    'ALTER TABLE {schema}.image_user DROP CONSTRAINT image_user_image_fkey;',
    'ALTER TABLE {schema}.image_user ADD CONSTRAINT image_user_image_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE {schema}.annotation DROP CONSTRAINT annotation_username_fkey;',
    'ALTER TABLE {schema}.annotation ADD CONSTRAINT annotation_username_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE {schema}.cnnstate ADD COLUMN IF NOT EXISTS model_library VARCHAR',
    'ALTER TABLE {schema}.cnnstate ADD COLUMN IF NOT EXISTS alCriterion_library VARCHAR',
    'ALTER TABLE {schema}.image ADD COLUMN IF NOT EXISTS isGoldenQuestion BOOLEAN NOT NULL DEFAULT FALSE',
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
    '''
]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update AIde database structure.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

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
    if not len(alCriterionPath): alCriterionPath = None

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


    # register project
    secretToken = secrets.token_urlsafe(32)
    dbConn.execute('''
        INSERT INTO aide_admin.project (shortname, name, description,
            secret_token,
            interface_enabled,
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
            %s, %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (shortname) DO NOTHING;
    ''',
        (
            config.getProperty('Database', 'schema'),
            config.getProperty('Project', 'projectName'),
            config.getProperty('Project', 'projectDescription'),
            secretToken,
            True,
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


    # The multi-project AIde setup requires images to be in a subfolder named after
    # the project shorthand. Here we tell the user about moving the files, or else
    # propose a temporary fix (softlink).
    softlinkName = config.getProperty('FileServer', 'staticfiles_dir')
    if not os.path.isdir(softlinkName):
        # not running on file server; show message
        print('You do not appear to be running AIde on a "FileServer" instance.')
        print('INFO: In the process of AIde supporting multiple projects, each')
        print('project\'s files must be put in a sub-folder named after the project\'s')
        print('shorthand (i.e.: {}/{}/<images>).'.format(
            softlinkName,
            config.getProperty('Database', 'schema')))
        print('Make sure to run this script on all "FileServer" instance(s) as well')
        print('to address this issue.')
    
    else:
        softlinkName = os.path.join(softlinkName, config.getProperty('Database', 'schema'))
        if os.path.islink(softlinkName):
            print('INFO: Detected link to project file directory ({})'.format(softlinkName))
            print('You might want to move the files to a dedicated folder at some point...')
        else:
            print('INFO: In the process of AIde supporting multiple projects, each')
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

    # ask before inserting existing user data
    print('Migration will now transfer the existing user accounts for project "{projectName}" to the new schema.'.format(
        projectName=config.getProperty('Project', 'projectName')
    ))
    print('WARNING: any account whose name is already in the new schema will be ignored.')
    print('This means that users with accounts on multiple projects will have to log in')
    print('with the password they used for the first project that has been migrated.')
    print('\nWould you like to migrate the users automatically now?')
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
        dbConn.execute('''
                INSERT INTO aide_admin.user (name, email, hash, last_login, session_token)
                SELECT name, email, hash, last_login, session_token
                FROM {schema}.user
                ON CONFLICT (name) DO NOTHING;
            '''.format(schema=config.getProperty('Database', 'schema')),
            None,
            None
        )


    # add authentication
    dbConn.execute('''
            INSERT INTO aide_admin.authentication (username, project, isAdmin)
            SELECT name, '{schema}', isAdmin FROM {schema}.user
            WHERE name IN (SELECT name FROM aide_admin.user)
            ON CONFLICT (username, project) DO NOTHING;
        '''.format(schema=config.getProperty('Database', 'schema')),
        None,
        None)

    print('Project "{}" is now up-to-date for the latest changes in AIde.'.format(config.getProperty('Project', 'projectName')))