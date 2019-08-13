'''
    Script to establish a database schema according to the specifications
    provided in the configuration file.

    2019 Benjamin Kellenberger
'''

import os
import argparse
from util.configDef import Config
from modules import Database, UserHandling


def _constructAnnotationFields(annoType, table, doublePrecision=False):
    coordType = 'real'
    if doublePrecision:
        coordType = 'double precision'

    if False:       #TODO: separate confidence values for all labelclasses? How to handle empty class? (table == 'prediction'):
        confString = ''
        if table == 'prediction':
            confString = 'conidence real,'
        additionalTables = '''CREATE TABLE IF NOT EXISTS &schema.PREDICTION_LABELCLASS (
            prediction uuid NOT NULL,
            labelclass uuid NOT NULL,
            {confString}
            PRIMARY KEY (prediction, labelclass),
            FOREIGN KEY (prediction) REFERENCES &schema.PREDICTION(id),
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id)
        );
        '''.format(confString=confString)
    else:
        additionalTables = None

    if annoType == 'labels':
        annoFields = '''
            label uuid &labelclassNotNull,
            FOREIGN KEY (label) REFERENCES &schema.LABELCLASS(id),
        '''
    
    elif annoType == 'points':
        annoFields = '''
            label uuid &labelclassNotNull,
            x {coordType},
            y {coordType},
            FOREIGN KEY (label) REFERENCES &schema.LABELCLASS(id),
        '''.format(coordType=coordType)

    elif annoType == 'boundingBoxes':
        annoFields = '''
            label uuid &labelclassNotNull,
            x {coordType},
            y {coordType},
            width {coordType},
            height {coordType},
            FOREIGN KEY (label) REFERENCES &schema.LABELCLASS(id),
        '''.format(coordType=coordType)

    elif annoType == 'segmentationMasks':
        additionalTables = None     # not needed for semantic segmentation
        annoFields = '''
            segmentationMask VARCHAR,
            width INTEGER,
            height INTEGER,
        '''

    else:
        raise ValueError('{} is not a recognized annotation type.'.format(annoType))
    
    return annoFields, additionalTables



if __name__ == '__main__':

    # setup
    parser = argparse.ArgumentParser(description='Run CV4Wildlife AL Service.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Directory of the settings.ini file used for this machine (default: "config/settings.ini").')
    args = parser.parse_args()

    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = args.settings_filepath

    config = Config()
    dbConn = Database(config)


    # read SQL skeleton
    with open(os.path.join(os.getcwd(), 'projectCreation/db_create.sql'), 'r') as f:
        sql = f.read()
    
    # fill in placeholders
    annoType_frontend = config.getProperty('Project', 'annotationType')
    annoFields_frontend, additionalTables_frontend = _constructAnnotationFields(annoType_frontend, 'annotation')
    if additionalTables_frontend is not None:
        sql += additionalTables_frontend

    annoType_backend = config.getProperty('Project', 'predictionType')
    annoFields_backend, additionalTables_backend = _constructAnnotationFields(annoType_backend, 'prediction')
    if additionalTables_backend is not None:
        sql += additionalTables_backend

    sql = sql.replace('&annotationFields', annoFields_frontend)
    sql = sql.replace('&predictionFields', annoFields_backend)
    
    sql = sql.replace('&dbName', config.getProperty('Database', 'name'))
    sql = sql.replace('&owner', config.getProperty('Database', 'user'))     #TODO
    sql = sql.replace('&user', config.getProperty('Database', 'user'))
    sql = sql.replace('&password', config.getProperty('Database', 'password'))
    sql = sql.replace('&schema', config.getProperty('Database', 'schema'))

    requireLabelClass = ('' if config.getProperty('Project', 'enableEmptyClass', type=bool, fallback=False) else 'NOT NULL')
    sql = sql.replace('&labelclassNotNull', requireLabelClass)


    # run SQL
    dbConn.execute(sql, None, None)


    # add admin user
    sql = '''
        INSERT INTO {}.user (name, email, hash, isadmin)
        VALUES (%s, %s, %s, %s)
    '''.format(config.getProperty('Database', 'schema'))

    adminPass = config.getProperty('Project', 'adminPassword')
    uHandler = UserHandling.backend.middleware.UserMiddleware(config)
    adminPass = uHandler._create_hash(adminPass.encode('utf8'))

    values = (config.getProperty('Project', 'adminName'), config.getProperty('Project', 'adminEmail'), adminPass, True,)

    dbConn.execute(sql, values, None)