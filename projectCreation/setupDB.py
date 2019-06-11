'''
    Script to establish a database schema according to the specifications
    provided in the configuration file.

    2019 Benjamin Kellenberger
'''

import os
from util.configDef import Config
from modules import Database


def _constructAnnotationFields(annoType, table):
    if table == 'prediction':
        additionalTables = '''CREATE TABLE IF NOT EXISTS &schema.PREDICTION_LABELCLASS (
            predictionID uuid NOT NULL,
            labelclassID uuid NOT NULL,
            confidence real,
            PRIMARY KEY (predictionID, labelclassID),
            FOREIGN KEY (predictionID) REFERENCES &schema.PREDICTION(id),
            FOREIGN KEY (labelclassID) REFERENCES &schema.LABELCLASS(id)
        );
        '''
    else:
        additionalTables = None

    if annoType == 'classLabels':
        annoFields = '''
            labelclass uuid NOT NULL,
            confidence real,
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''
    
    elif annoType == 'points':
        annoFields = '''
            labelclass uuid NOT NULL,
            confidence real,
            x integer,
            y integer,
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''

    elif annoType == 'boundingBoxes':
        annoFields = '''
            labelclass uuid NOT NULL,
            confidence real,
            x integer,
            y integer,
            width integer,
            height integer,
            FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id),
        '''

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


    # run SQL
    dbConn.execute(sql, None, None)