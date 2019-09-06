'''
    Run this file whenever you update AIde to bring your existing project setup up-to-date
    with respect to changes due to newer versions.
    
    2019 Benjamin Kellenberger
'''

import os
import argparse


MODIFICATIONS_sql = [
    'ALTER TABLE {schema}.annotation ADD COLUMN IF NOT EXISTS meta VARCHAR; ALTER TABLE {schema}.image_user ADD COLUMN IF NOT EXISTS meta VARCHAR;',
    'ALTER TABLE {schema}.labelclass ADD COLUMN IF NOT EXISTS keystroke SMALLINT UNIQUE;'
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

    print('Project {} is now up-to-date for the latest changes in AIde.'.format(config.getProperty('Project', 'projectName')))