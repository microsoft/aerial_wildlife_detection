'''
    Database connection functionality.

    2019-21 Benjamin Kellenberger
'''

from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, execute_values
psycopg2.extras.register_uuid()
from util.helpers import LogDecorator


class Database():

    def __init__(self, config, verbose_start=False):
        self.config = config

        if verbose_start:
            print('Connecting to database...'.ljust(LogDecorator.get_ljust_offset()), end='')

        # get DB parameters
        try:
            self.database = config.getProperty('Database', 'name').lower()
            self.host = config.getProperty('Database', 'host')
            self.port = config.getProperty('Database', 'port')
            self.user = config.getProperty('Database', 'user').lower()
            self.password = config.getProperty('Database', 'password')
        except Exception as e:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Incomplete database credentials provided in configuration file (message: "{str(e)}").')

        try:
            self._createConnectionPool()
        except Exception as e:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not connect to database (message: "{str(e)}").')

        if verbose_start:
            LogDecorator.print_status('ok')


    def __del__(self):
        try:
            self.connectionPool.closeall()
        except:
            pass


    def _createConnectionPool(self):
        self.connectionPool = ThreadedConnectionPool(
            1,
            self.config.getProperty('Database', 'max_num_connections', type=int, fallback=20),
            host=self.host,
            database=self.database,
            port=self.port,
            user=self.user,
            password=self.password,
            connect_timeout=2
        )


    def runServer(self):
        ''' Dummy function for compatibility reasons '''
        return



    @contextmanager
    def _get_connection(self):
        conn = self.connectionPool.getconn()
        conn.autocommit = True
        try:
            yield conn
        finally:
            self.connectionPool.putconn(conn, close=False)


    def execute(self, query, arguments, numReturn=None):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # execute statement
                try:
                    cursor.execute(query, arguments)
                    conn.commit()

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
                    except Exception as eGet:
                        print(eGet)

                except Exception as e:
                    if not conn.closed:
                        conn.rollback()
                    print(e)
                    # self.connectionPool.putconn(conn, close=False)    #TODO: this still causes connection to close
                    
                    # conn = self.connectionPool.getconn()

                    # # retry execution
                    # try:
                    #     cursor = conn.cursor(cursor_factory=RealDictCursor)
                    #     cursor.execute(query, arguments)
                    #     conn.commit()
                    # except:
                    #     if not conn.closed:
                    #         conn.rollback()
                    #     print(e)
        

    # def execute_cursor(self, query, arguments):
    #     with self._get_connection() as conn:
    #         with conn.cursor(cursor_factory=RealDictCursor) as cursor:
    #             try:
    #                 cursor.execute(query, arguments)
    #                 conn.commit()
    #                 yield cursor
    #             except Exception as e:
    #                 if not conn.closed:
    #                     conn.rollback()
    #                 print(e)
    #                 # cursor.close()

    #                 # # retry execution
    #                 # conn = self.connectionPool.getconn()
    #                 # try:
    #                 #     cursor = conn.cursor(cursor_factory=RealDictCursor)
    #                 #     cursor.execute(query, arguments)
    #                 #     conn.commit()
    #                 # except Exception as e:
    #                 #     if not conn.closed:
    #                 #         conn.rollback()
    #                 #     print(e)
    #             finally:
    #                 if not cursor.closed:
    #                     cursor.close()


    def insert(self, query, values, numReturn=None):
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    execute_values(cursor, query, values)
                    conn.commit()

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
                    except Exception as eGet:
                        print(eGet)

                except Exception as e:
                    if not conn.closed:
                        conn.rollback()
                    print(e)
                    # cursor.close()

                    # # retry execution
                    # conn = self.connectionPool.getconn()
                    # try:
                    #     cursor = conn.cursor(cursor_factory=RealDictCursor)
                    #     execute_values(cursor, query, values)
                    #     conn.commit()
                    # except Exception as e:
                    #     if not conn.closed:
                    #         conn.rollback()
                    #     print(e)
                finally:
                    if not cursor.closed:
                        cursor.close()