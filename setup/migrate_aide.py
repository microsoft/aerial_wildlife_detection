'''
    Run this file whenever you update AIDE to bring your existing project setup up-to-date
    with respect to changes due to newer versions.
    
    2019-21 Benjamin Kellenberger
'''

import os
import argparse
from psycopg2 import sql
from constants import version


MODIFICATIONS_sql = [
    'ALTER TABLE "{schema}".annotation ADD COLUMN IF NOT EXISTS meta VARCHAR; ALTER TABLE "{schema}".image_user ADD COLUMN IF NOT EXISTS meta VARCHAR;',
    'ALTER TABLE "{schema}".labelclass ADD COLUMN IF NOT EXISTS keystroke SMALLINT UNIQUE;',
    'ALTER TABLE "{schema}".image ADD COLUMN IF NOT EXISTS last_requested TIMESTAMPTZ;',

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
    'ALTER TABLE "{schema}".image_user DROP CONSTRAINT IF EXISTS image_user_image_fkey;',
    'ALTER TABLE "{schema}".image_user DROP CONSTRAINT IF EXISTS image_user_username_fkey;',
    '''DO
        $do$
        BEGIN
        IF EXISTS (
            SELECT 1
            FROM   information_schema.tables 
            WHERE  table_schema = '"{schema}"'
            AND    table_name = 'user'
        ) THEN
            INSERT INTO aide_admin.user (name, email, hash, isSuperUser, canCreateProjects, secret_token)
            SELECT name, email, hash, false AS isSuperUser, false AS canCreateProjects, md5(random()::text) AS secret_token FROM "{schema}".user
            ON CONFLICT(name) DO NOTHING;
        END IF;
    END $do$;''',
    'ALTER TABLE "{schema}".image_user ADD CONSTRAINT image_user_image_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE "{schema}".annotation DROP CONSTRAINT IF EXISTS annotation_username_fkey;',
    'ALTER TABLE "{schema}".annotation ADD CONSTRAINT annotation_username_fkey FOREIGN KEY (username) REFERENCES aide_admin.USER (name);',
    'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS model_library VARCHAR;',
    'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS alCriterion_library VARCHAR;',
    'ALTER TABLE "{schema}".image ADD COLUMN IF NOT EXISTS isGoldenQuestion BOOLEAN NOT NULL DEFAULT FALSE;',
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
    'ALTER TABLE "{schema}".image ADD COLUMN IF NOT EXISTS date_added TIMESTAMPTZ NOT NULL DEFAULT NOW();',
    'ALTER TABLE aide_admin.authentication ADD COLUMN IF NOT EXISTS admitted_until TIMESTAMPTZ;',
    'ALTER TABLE aide_admin.authentication ADD COLUMN IF NOT EXISTS blocked_until TIMESTAMPTZ;',
    #TODO: we probably don't need unique group names (useful e.g. for nested groups)
    # '''ALTER TABLE "{schema}".labelclassgroup DROP CONSTRAINT IF EXISTS labelclassgroup_name_unique;
    # ALTER TABLE "{schema}".labelclassgroup ADD CONSTRAINT labelclassgroup_name_unique UNIQUE (name);''',
    'ALTER TABLE "{schema}".labelclassgroup DROP CONSTRAINT IF EXISTS labelclassgroup_name_unique;',
    'ALTER TABLE "{schema}".image ADD COLUMN IF NOT EXISTS corrupt BOOLEAN;',
    '''
    CREATE TABLE IF NOT EXISTS "{schema}".workflow (
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
    'ALTER TABLE "{schema}".image_user ADD COLUMN IF NOT EXISTS num_interactions INTEGER NOT NULL DEFAULT 0;'
    'ALTER TABLE "{schema}".image_user ADD COLUMN IF NOT EXISTS first_checked TIMESTAMPTZ;',
    'ALTER TABLE "{schema}".image_user ADD COLUMN IF NOT EXISTS total_time_required BIGINT;',
    '''ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS segmentation_ignore_unlabeled BOOLEAN NOT NULL DEFAULT TRUE;
        ALTER TABLE "{schema}".labelclass ADD COLUMN IF NOT EXISTS hidden BOOLEAN NOT NULL DEFAULT FALSE;
    ''',
    'ALTER TABLE "{schema}".annotation ADD COLUMN IF NOT EXISTS autoConverted BOOLEAN;',
    'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS owner VARCHAR;',
    '''ALTER TABLE aide_admin.project DROP CONSTRAINT IF EXISTS project_user_fkey;
        ALTER TABLE aide_admin.project ADD CONSTRAINT project_user_fkey FOREIGN KEY (owner) REFERENCES aide_admin.USER (name);''',
    
    # new workflow history
    ''' CREATE TABLE IF NOT EXISTS "{schema}".workflowHistory (
        id uuid DEFAULT uuid_generate_v4(),
        workflow VARCHAR NOT NULL,
        tasks VARCHAR,
        timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        timeFinished TIMESTAMPTZ,
        launchedBy VARCHAR,
        abortedBy VARCHAR,
        succeeded BOOLEAN,
        messages VARCHAR,
        PRIMARY KEY (id),
        FOREIGN KEY (launchedBy) REFERENCES aide_admin.user (name),
        FOREIGN KEY (abortedBy) REFERENCES aide_admin.user (name)
    );''',

    # project folder watching
    'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS watch_folder_enabled BOOLEAN NOT NULL DEFAULT FALSE;',
    'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS watch_folder_remove_missing_enabled BOOLEAN NOT NULL DEFAULT FALSE;',

    # model marketplace
    '''CREATE TABLE IF NOT EXISTS aide_admin.modelMarketplace (
        id uuid DEFAULT uuid_generate_v4(),
        name VARCHAR UNIQUE NOT NULL,
        description VARCHAR NOT NULL,
        labelclasses VARCHAR NOT NULL,
        author VARCHAR NOT NULL,
        model_library VARCHAR NOT NULL,
        statedict BYTEA,
        timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        alCriterion_library VARCHAR,
        origin_project VARCHAR,
        origin_uuid UUID,
        origin_uri VARCHAR,
        public BOOLEAN NOT NULL DEFAULT TRUE,
        anonymous BOOLEAN NOT NULL DEFAULT FALSE,
        selectCount INTEGER NOT NULL DEFAULT 0,
        shared BOOLEAN NOT NULL DEFAULT TRUE,
        tags VARCHAR,
        PRIMARY KEY (id)
    );''',
    'ALTER TABLE aide_admin.modelMarketplace ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT TRUE;',
    'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS marketplace_origin_id UUID;',
    'ALTER TABLE "{schema}".cnnstate DROP CONSTRAINT IF EXISTS marketplace_origin_id_fkey;'
    'ALTER TABLE "{schema}".cnnstate ADD CONSTRAINT marketplace_origin_id_fkey FOREIGN KEY (marketplace_origin_id) REFERENCES aide_admin.modelMarketplace(id);',
    'ALTER TABLE aide_admin.modelMarketplace ADD COLUMN IF NOT EXISTS tags VARCHAR;',
    'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS archived BOOLEAN DEFAULT FALSE;',
    '''
        /*
            Last occurrence of substring. Function obtained from here:
            https://wiki.postgresql.org/wiki/Strposrev
        */
        CREATE OR REPLACE FUNCTION strposrev(instring text, insubstring text)
        RETURNS integer AS
        $BODY$
        DECLARE result INTEGER;
        BEGIN
            IF strpos(instring, insubstring) = 0 THEN
            -- no match
            result:=0;
            ELSEIF length(insubstring)=1 THEN
            -- add one to get the correct position from the left.
            result:= 1 + length(instring) - strpos(reverse(instring), insubstring);
            ELSE 
            -- add two minus the legth of the search string
            result:= 2 + length(instring)- length(insubstring) - strpos(reverse(instring), reverse(insubstring));
            END IF;
            RETURN result;
        END;
        $BODY$
        LANGUAGE plpgsql IMMUTABLE STRICT
        COST 4;
  ''',
  '''
    CREATE OR REPLACE VIEW "{schema}".fileHierarchy AS (
        SELECT DISTINCT
        CASE WHEN position('/' IN filename) = 0 THEN null
        ELSE left(filename, strposrev(filename, '/')-1) END
        AS folder
        FROM "{schema}".image
    );
  ''',
  '''
    CREATE TABLE IF NOT EXISTS "{schema}".bookmark (
        username VARCHAR NOT NULL,
        image uuid NOT NULL,
        timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (username, image),
        FOREIGN KEY (username) REFERENCES "aide_admin".user(name),
        FOREIGN KEY (image) REFERENCES "{schema}".image
    );
  ''',

#   # change in workflow definition #TODO: takes forever...
#   '''
#     DELETE FROM "{schema}".workflowhistory
#     WHERE timecreated < TO_DATE('20201204','YYYYMMDD');
#   ''',

  '''
    ALTER TABLE "{schema}".labelclass ADD COLUMN IF NOT EXISTS
    timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW();
  ''',

  f'''
    CREATE TABLE IF NOT EXISTS "aide_admin".version (
        version VARCHAR UNIQUE NOT NULL,
        PRIMARY KEY (version)
    );
  ''',
  'ALTER TABLE aide_admin.modelMarketplace ADD COLUMN IF NOT EXISTS origin_uri VARCHAR UNIQUE;',

  'ALTER TABLE aide_admin.modelMarketplace ALTER stateDict DROP NOT NULL;',  # due to pre-trained models we now allow empty state dicts)...
  'ALTER TABLE aide_admin.modelMarketplace DROP CONSTRAINT IF EXISTS modelmarketplace_author_fkey;', # ...as well as foreign model authors
  'ALTER TABLE "{schema}".cnnstate ALTER stateDict DROP NOT NULL;',
  '''
    CREATE TABLE IF NOT EXISTS "{schema}".taskhistory (
        id uuid NOT NULL DEFAULT uuid_generate_v4(),
        task_id VARCHAR NOT NULL,
        launchedBy VARCHAR,
        abortedBy VARCHAR,
        processDescription VARCHAR,
        timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        timeFinished TIMESTAMPTZ,
        result VARCHAR,
        PRIMARY KEY (id),
        FOREIGN KEY (launchedBy) REFERENCES aide_admin.user (name),
        FOREIGN KEY (abortedBy) REFERENCES aide_admin.user (name)
    );
  ''',
  'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS stats VARCHAR;',
  'ALTER TABLE aide_admin.modelMarketplace ADD COLUMN IF NOT EXISTS model_settings VARCHAR;',
  'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS inference_chunk_size BIGINT;',
  'ALTER TABLE aide_admin.project ADD COLUMN IF NOT EXISTS max_num_concurrent_tasks INTEGER;',

  # explicit model-to-labelclass mapping
  '''
    CREATE TABLE IF NOT EXISTS "{schema}".model_labelclass (
        --ai_model_library VARCHAR NOT NULL,
        marketplace_origin_id UUID NOT NULL,
        labelclass_id_model VARCHAR NOT NULL,
        labelclass_name_model VARCHAR NOT NULL,
        labelclass_id_project UUID,
        PRIMARY KEY (ai_model_library, labelclass_id_model),
        FOREIGN KEY (labelclass_id_project) REFERENCES "{schema}".labelclass (id)
    );
  ''',
  'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS imported_from_marketplace BOOLEAN NOT NULL DEFAULT FALSE;',
  'ALTER TABLE "{schema}".cnnstate ADD COLUMN IF NOT EXISTS labelclass_autoupdate BOOLEAN NOT NULL DEFAULT FALSE;',
  'ALTER TABLE "aide_admin".project ADD COLUMN IF NOT EXISTS labelclass_autoupdate BOOLEAN NOT NULL DEFAULT FALSE;',

  # thanks to "imported_from_marketplace" field, we don't want the unique constraint on the origin ID anymore
  'ALTER TABLE "{schema}".cnnstate DROP CONSTRAINT IF EXISTS cnnstate_marketplace_origin_id_key;',

  # we also allow multiple models with the same origin URI (for updates of Web-imported models or uploads with iid file name)
  'ALTER TABLE "aide_admin".modelmarketplace DROP CONSTRAINT IF EXISTS modelmarketplace_origin_uri_key;'
]



def migrate_aide(forceMigrate=False):
    from modules import Database, UserHandling
    from util.configDef import Config
    
    config = Config()
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')
    
    warnings = []
    errors = []

    # skip if not forced and if database has same version
    doMigrate = True
    
    # check if DB has version already implemented
    dbVersion = None
    hasVersion = dbConn.execute('''
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE  table_schema = 'aide_admin'
            AND    table_name   = 'version'
        ) AS hasVersion;
    ''', None, 1)
    if hasVersion[0]['hasversion']:
        # check DB version
        dbVersion = dbConn.execute('SELECT version FROM aide_admin.version;', None, 1)
        if dbVersion is not None and len(dbVersion):
            dbVersion = dbVersion[0]['version']
            needsUpdate = version.compare_versions(version.AIDE_VERSION, dbVersion)
            if needsUpdate is not None:
                if needsUpdate < 0:
                    # running an older version of AIDE with a newer DB version
                    warnings.append(f'WARNING: local AIDE version ({version.AIDE_VERSION}) is older than the one in the database ({dbVersion}); please update your installation.')
                elif needsUpdate == 0:
                    doMigrate = False
                else:
                    doMigrate = True

    if not doMigrate and not forceMigrate:
        return warnings, errors

    # bring all projects up-to-date (if registered within AIDE)
    projects = dbConn.execute('SELECT shortname FROM aide_admin.project;', None, 'all')
    if projects is not None and len(projects):

        # get all schemata and check if project still exists
        schemata = dbConn.execute('SELECT schema_name FROM information_schema.schemata', None, 'all')
        if schemata is not None and len(schemata):
            schemata = set([s['schema_name'].lower() for s in schemata])
            for p in projects:
                try:
                    pName = p['shortname']

                    # check if project still exists
                    if not pName.lower() in schemata:
                        warnings.append(f'WARNING: project "{pName}" is registered but does not exist in database.')
                        #TODO: option to auto-remove?
                        continue

                    # special modification for CNN-to-labelclass map: drop only dep. on version (remove ancient tests)
                    if version.compare_versions(version.AIDE_VERSION, dbVersion) in (-1, None):
                        dbConn.execute(sql.SQL('DROP TABLE IF EXISTS {};').format(
                            sql.Identifier(pName, 'cnn_labelclass')
                        ), None)

                    # make modifications one at a time
                    for mod in MODIFICATIONS_sql:
                        dbConn.execute(mod.format(schema=pName), None, None)

                    # pre-official 2.0: mark existing CNN states as "labelclass_autoupdate" (as this was the default behavior)
                    if version.compare_versions(dbVersion, '2.0.210514') == -1:
                        dbConn.execute(sql.SQL('''
                            UPDATE {}
                            SET labelclass_autoupdate = TRUE;
                        ''').format(sql.Identifier(pName, 'cnnstate')), None)

                except Exception as e:
                    errors.append(str(e))
        else:
            warnings.append('WARNING: no project schemata found within database.')
    else:
        warnings.append('WARNING: no project registered within AIDE.')

    # update DB version accordingly
    dbConn.execute('''
        DELETE FROM aide_admin.version;
        INSERT INTO aide_admin.version (version)
        VALUES (%s);
    ''', (version.AIDE_VERSION, ))

    return warnings, errors
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update AIDE database structure.')
    parser.add_argument('--force', type=int, default=0,
                    help='Set to 1 to force migration, even if AIDE versions already match.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)
    if not 'AIDE_MODULES' in os.environ:
        os.environ['AIDE_MODULES'] = ''     # for compatibility with Celery worker import

    
    warnings, errors = migrate_aide(args.force)

    if not len(warnings) and not len(errors):
        print(f'AIDE is now up-to-date with the latest version ({version.AIDE_VERSION})')
    else:
        print(f'Warnings and/or errors occurred while updating AIDE to the latest version ({version.AIDE_VERSION}):')
        if len(warnings):
            print('\nWarnings:')
            for w in warnings:
                print(f'\t"{w}"')
            
        if len(errors):
            print('\nErrors:')
            for e in errors:
                print(f'\t"{e}"')