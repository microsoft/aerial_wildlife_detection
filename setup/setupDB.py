'''
    Script to establish a database schema according to the specifications
    provided in the configuration file. Unlike in previous versions of
    AIDE, this does not set up any project-specific schemata anymore.
    This task is achieved through the GUI.
    See modules.ProjectAdministration.backend.middleware.py for details.

    2019-21 Benjamin Kellenberger
'''

import os

if not 'AIDE_MODULES' in os.environ:
    os.environ['AIDE_MODULES'] = 'FileServer'     # for compatibility with Celery worker import

import argparse
from constants.version import AIDE_VERSION
from util.configDef import Config
from modules import Database, UserHandling
from setup.migrate_aide import migrate_aide



def add_update_superuser(config, dbConn):
    '''
        Reads the super user credentials from the config file and checks if
        anything has changed w.r.t. the entries in the database. Makes
        modifications if this is the case and reports back the changes.
    '''
    isNewAccount = False
    changes = {}

    # values in config file
    adminName = config.getProperty('Project', 'adminName')
    if adminName is None or not len(adminName):
        return None
    adminEmail = config.getProperty('Project', 'adminEmail')
    adminPass = config.getProperty('Project', 'adminPassword')
    if adminPass is None or not len(adminPass):
        raise Exception('No password defined for admin account in configuration file.')
    uHandler = UserHandling.backend.middleware.UserMiddleware(config, dbConn)
    adminPass = uHandler._create_hash(adminPass.encode('utf8'))

    # get current values
    currentMeta = dbConn.execute('''
        SELECT email, hash
        FROM aide_admin.user
        WHERE name = %s;
    ''', (adminName,), 1)
    if currentMeta is None or not len(currentMeta):
        # no account found under this name; create new
        isNewAccount = True
    
    # check if changes
    if currentMeta is not None and len(currentMeta):
        currentMeta = currentMeta[0]
        if currentMeta['email'] != adminEmail:
            changes['adminEmail'] = True
        if bytes(currentMeta['hash']) != adminPass:
            changes['adminPassword'] = True

    if isNewAccount or len(changes):
        sql = '''
            INSERT INTO aide_admin.user (name, email, hash, issuperuser)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET email=EXCLUDED.email, hash=EXCLUDED.hash;
        '''
        values = (adminName, adminEmail, adminPass, True,)
        dbConn.execute(sql, values, None)

    return {
        'details': {
            'name': adminName,
            'email': adminEmail
        },
        'new_account': isNewAccount,
        'changes': changes
    }



def setupDB():
    config = Config()
    dbConn = Database(config)

    # read SQL skeleton
    with open(os.path.join(os.getcwd(), 'setup/db_create.sql'), 'r') as f:
        sql = f.read()
    
    # fill in placeholders
    sql = sql.replace('&user', config.getProperty('Database', 'user'))

    # run SQL
    dbConn.execute(sql, None, None)

    # add admin user
    add_update_superuser(config, dbConn)

    # finalize: migrate database in any case (this also adds the AIDE version if needed)
    migrate_aide()

    # fresh database; add AIDE version
    dbConn.execute('''
        INSERT INTO aide_admin.version (version)
        VALUES (%s)
        ON CONFLICT (version) DO NOTHING;
    ''', (AIDE_VERSION,), None)



if __name__ == '__main__':

    # setup
    parser = argparse.ArgumentParser(description='Set up AIDE database schema.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = args.settings_filepath
    
    setupDB()