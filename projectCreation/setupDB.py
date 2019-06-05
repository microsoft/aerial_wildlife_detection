'''
    Script to establish a database schema according to the specifications
    provided in the configuration file.

    2019 Benjamin Kellenberger
'''

import os
from util.configDef import Config
from modules import Database



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
    sql = sql.replace('&dbName', config.getProperty('DATABASE', 'name'))
    sql = sql.replace('&owner', config.getProperty('DATABASE', 'user'))     #TODO
    sql = sql.replace('&user', config.getProperty('DATABASE', 'user'))
    sql = sql.replace('&password', config.getProperty('DATABASE', 'password'))
    sql = sql.replace('&schema', config.getProperty('DATABASE', 'schema'))

    # run SQL
    dbConn.execute(sql)