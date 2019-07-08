'''
    Database connection functionality.

    2019 Benjamin Kellenberger
'''

from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, execute_values
psycopg2.extras.register_uuid()



class Database():

    def __init__(self, config):
        self.config = config

        # get DB parameters
        self.database = config.getProperty('Database', 'name').lower()
        self.host = config.getProperty('Database', 'host')
        self.port = config.getProperty('Database', 'port')
        self.user = config.getProperty('Database', 'user').lower()
        self.password = config.getProperty('Database', 'password')

        self._createConnectionPool()


    def _createConnectionPool(self):
        try:
            self.connectionPool = ThreadedConnectionPool(
                1,
                self.config.getProperty('Database', 'max_num_connections', type=int, fallback=20),
                host=self.host,
                database=self.database,
                port=self.port,
                user=self.user,
                password=self.password
            )
        except Exception as e:
            print(e)
            print('Error connecting to database {}:{} with username {}.'.format(
                self.host, self.port, self.user
            ))
            #TODO: next steps


    def runServer(self):
        ''' Dummy function for compatibility reasons '''
        return



    @contextmanager
    def _get_connection(self):
        conn = self.connectionPool.getconn()
        try:
            yield conn
        finally:
            self.connectionPool.putconn(conn, close=False)


    def execute(self, sql, arguments, numReturn=None):
        # with self.connectionPool.getconn() as conn:
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory = RealDictCursor)

            # execute statement
            try:
                cursor.execute(sql, arguments)
                conn.commit()
            except Exception as e:
                print(e)
                conn.rollback()

            # get results
            try:
                returnValues = []
                if numReturn is None:
                    # cursor.close()
                    return
                
                elif numReturn == 'all':
                    returnValues = cursor.fetchall()
                    # cursor.close()
                    return returnValues

                else:
                    for _ in range(numReturn):
                        rv = cursor.fetchone()
                        if rv is None:
                            return returnValues
                        returnValues.append(rv)
        
                    # cursor.close()
                    return returnValues
            except Exception as e:
                print(e)
                # conn.rollback()
            finally:
                pass
                # self.connectionPool.putconn(conn)
                # cursor.close()
    

    def execute_cursor(self, sql, arguments):
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory = RealDictCursor)
            try:
                cursor.execute(sql, arguments)
                conn.commit()
                return cursor
            except Exception as e:
                print(e)
                conn.rollback()


    def insert(self, sql, values):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(cursor, sql, values)
                conn.commit()
            except Exception as e:
                print(e)
                conn.rollback()
            finally:
                pass
                # cursor.close()