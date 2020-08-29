'''
    Script to establish a database schema according to the specifications
    provided in the configuration file. Unlike in previous versions of
    AIDE, this does not set up any project-specific schemata anymore.
    This task is achieved through the GUI.
    See modules.ProjectAdministration.backend.middleware.py for details.

    2019-20 Benjamin Kellenberger
'''

import os

if not 'AIDE_MODULES' in os.environ:
    os.environ['AIDE_MODULES'] = 'FileServer'     # for compatibility with Celery worker import

import argparse
from util.configDef import Config
from modules import Database, UserHandling


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
    sql = '''
        INSERT INTO aide_admin.user (name, email, hash, issuperuser)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name) DO NOTHING;
    '''
    adminPass = config.getProperty('Project', 'adminPassword')
    uHandler = UserHandling.backend.middleware.UserMiddleware(config)
    adminPass = uHandler._create_hash(adminPass.encode('utf8'))

    values = (config.getProperty('Project', 'adminName'), config.getProperty('Project', 'adminEmail'), adminPass, True,)
    dbConn.execute(sql, values, None)


if __name__ == '__main__':

    # setup
    parser = argparse.ArgumentParser(description='Set up AIDE database schema.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = args.settings_filepath
    
    setupDB()