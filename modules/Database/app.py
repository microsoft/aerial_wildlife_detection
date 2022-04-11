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
            self.user = config.getProperty('Database', 'user', fallback=None)
            self.password = config.getProperty('Database', 'password', fallback=None)

            if self.user is None or self.password is None:
                # load from credentials file instead
                credentials = config.getProperty('Database', 'credentials')
                with open(credentials, 'r') as c:
                    lines = c.readlines()
                    for line in lines:
                        line = line.lstrip().rstrip('\r').rstrip('\n')
                        if line.startswith('#') or line.startswith(';'):
                            continue
                        tokens = line.split('=')
                        if len(tokens) >= 2:
                            idx = line.find('=') + 1
                            field = tokens[0].strip().lower()
                            if field == 'username':
                                self.user = line[idx:]
                            elif field == 'password':
                                self.password = line[idx:]

            # self.user = self.user.lower()
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



    def _createConnectionPool(self):
        self.connectionPool = ThreadedConnectionPool(
            0,
            max(2, self.config.getProperty('Database', 'max_num_connections', type=int, fallback=20)),  # 2 connections are needed as minimum for retrying of execution
            host=self.host,
            database=self.database,
            port=self.port,
            user=self.user,
            password=self.password,
            connect_timeout=10
        )


    
    def canConnect(self):
        with self._get_connection() as conn:
            return conn is not None and not conn.closed



    @contextmanager
    def _get_connection(self):
        conn = self.connectionPool.getconn()
        conn.autocommit = True
        try:
            yield conn
        finally:
            self.connectionPool.putconn(conn)       #, close=False)



    def execute(self, query, arguments, numReturn=None):
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # execute statement
            try:
                cursor.execute(query, arguments)
                conn.commit()

                # get results
                returnValues = []
                if numReturn is None:
                    return None
                
                elif numReturn == 'all':
                    returnValues = cursor.fetchall()
                    return returnValues

                else:
                    for _ in range(numReturn):
                        rv = cursor.fetchone()
                        if rv is None:
                            return returnValues
                        returnValues.append(rv)
                    return returnValues

            except Exception as e:
                print(e)
                if not conn.closed:
                    conn.rollback()
            
            return None
                # # self.connectionPool.putconn(conn, close=False)    #TODO: this still causes connection to close
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

            # # get results
            # try:
            #     returnValues = []
            #     if numReturn is None:
            #         # cursor.close()
            #         return
                
            #     elif numReturn == 'all':
            #         returnValues = cursor.fetchall()
            #         # cursor.close()
            #         return returnValues

            #     else:
            #         for _ in range(numReturn):
            #             rv = cursor.fetchone()
            #             if rv is None:
            #                 return returnValues
            #             returnValues.append(rv)
        
            #         # cursor.close()
            #         return returnValues
            # except Exception as e:
            #     print(e)
    


    def insert(self, query, values, numReturn=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(cursor, query, values)
                conn.commit()

                # get results
                returnValues = []
                if numReturn is None:
                    # cursor.close()
                    return None
                
                elif numReturn == 'all':
                    returnValues = cursor.fetchall()
                    return returnValues

                else:
                    for _ in range(numReturn):
                        rv = cursor.fetchone()
                        if rv is None:
                            return returnValues
                        returnValues.append(rv)
                    return returnValues

            except Exception as e:
                print(e)
                if not conn.closed:
                    conn.rollback()
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
            
            # # get results
            # try:
            #     returnValues = []
            #     if numReturn is None:
            #         # cursor.close()
            #         return
                
            #     elif numReturn == 'all':
            #         returnValues = cursor.fetchall()
            #         # cursor.close()
            #         return returnValues

            #     else:
            #         for _ in range(numReturn):
            #             rv = cursor.fetchone()
            #             if rv is None:
            #                 return returnValues
            #             returnValues.append(rv)
        
            #         # cursor.close()
            #         return returnValues
            # except Exception as e:
            #     print(e)