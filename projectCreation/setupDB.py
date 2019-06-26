'''
    Script to establish a database schema according to the specifications
    provided in the configuration file.

    2019 Benjamin Kellenberger
'''

import os
from util.configDef import Config
from modules import Database, UserHandling


def _constructAnnotationFields(annoType, table, doublePrecision=False):
    coordType = 'real'
    if doublePrecision:
        coordType = 'double precision'

    if False:       #TODO: separate confidence values for all labelclasses? How to handle empty class? (table == 'prediction'):
        additionalTables = '''CREATE TABLE IF NOT EXISTS &schema.PREDICTION_LABELCLASS (
            prediction uuid NOT NULL,
            labelclass uuid NOT NULL,
            confidence real,
            PRIMARY KEY (prediction, labelclass),
            FOREIGN KEY (prediction) REFERENCES &schema.PREDICTION(id),
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id)
        );
        '''
    else:
        additionalTables = None

    if annoType == 'labels':
        annoFields = '''
            labelclass uuid &labelclassNotNull,
            confidence real,
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''
    
    elif annoType == 'points':
        annoFields = '''
            labelclass uuid &labelclassNotNull,
            confidence real,
            x {},
            y {},
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''.format(coordType, coordType)

    elif annoType == 'boundingBoxes':
        annoFields = '''
            labelclass uuid &labelclassNotNull,
            confidence real,
            x {},
            y {},
            width {},
            height {},
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''.format(coordType, coordType, coordType, coordType)

    elif annoType == 'segmentationMasks':
        additionalTables = None     # not needed for semantic segmentation
        annoFields = '''
            filename VARCHAR,
            FOREIGN KEY (labelclassID) REFERENCES &schema.LABELCLASS(id),
        '''
        raise NotImplementedError('Segmentation masks are not (yet) implemented.')

    else:
        raise ValueError('{} is not a recognized annotation type.'.format(annoType))
    
    return annoFields, additionalTables



if __name__ == '__main__':

    # setup
    config = Config()
    dbConn = Database(config)
    if dbConn.conn is None:
        raise Exception('Error connecting to database.')


    # read SQL skeleton
    with open(os.path.join(os.getcwd(), 'projectCreation/db_create.sql'), 'r') as f:
        sql = f.read()
    
    # fill in placeholders
    annoType_frontend = config.getProperty('LabelUI', 'annotationType')
    annoFields_frontend, additionalTables_frontend = _constructAnnotationFields(annoType_frontend, 'annotation')
    if additionalTables_frontend is not None:
        sql += additionalTables_frontend

    annoType_backend = config.getProperty('AITrainer', 'annotationType')
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

    adminPass = config.getProperty('Database', 'adminPassword')
    uHandler = UserHandling.backend.middleware.UserMiddleware(config)
    adminPass = uHandler._create_hash(adminPass.encode('utf8'))

    values = (config.getProperty('Database', 'adminName'), config.getProperty('Database', 'adminEmail'), adminPass, True,)

    dbConn.execute(sql, values, None)