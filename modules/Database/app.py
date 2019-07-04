'''
    Database connection functionality.

    2019 Benjamin Kellenberger
'''

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
psycopg2.extras.register_uuid()


class Database():

    def __init__(self, config):
        self.config = config

        # get DB parameters
        self.database = config.getProperty(self, 'name').lower()
        self.host = config.getProperty(self, 'host')
        self.port = config.getProperty(self, 'port')
        self.user = config.getProperty(self, 'user').lower()
        self.password = config.getProperty(self, 'password')

        self._createConnection()


    def _createConnection(self):
        try:
            self.conn = psycopg2.connect(host=self.host,
                                        database=self.database,
                                        port=self.port,
                                        user=self.user,
                                        password=self.password)
            self.conn.autocommit = True
        except:
            print('Error connecting to database {}:{} with username {}.'.format(
                self.host, self.port, self.user
            ))
            #TODO: next steps


    def runServer(self):
        ''' Dummy function for compatibility reasons '''
        return

    
    def execute(self, sql, arguments, numReturn=None):
        with self.execute_cursor(sql, arguments) as cursor:
            try:
                returnValues = []
                if numReturn is None:
                    cursor.close()
                    return
                
                elif numReturn == 'all':
                    returnValues = cursor.fetchall()
                    cursor.close()
                    return returnValues

                else:
                    for _ in range(numReturn):
                        rv = cursor.fetchone()
                        if rv is None:
                            return returnValues
                        returnValues.append(rv)
        
                    cursor.close()
                    return returnValues
            except Exception as e:
                print(e)
                self.conn.rollback()
            finally:
                cursor.close()
    

    def execute_cursor(self, sql, arguments):
        cursor = self.conn.cursor(cursor_factory = RealDictCursor)
        try:
            cursor.execute(sql, arguments)
            self.conn.commit()
            return cursor
        except Exception as e:
            print(e)
            self.conn.rollback()


    def insert(self, sql, values):
        cursor = self.conn.cursor()
        try:
            execute_values(cursor, sql, values)
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()
        finally:
            cursor.close()