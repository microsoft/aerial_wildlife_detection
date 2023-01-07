'''
    Database connection functionality.

    2019-23 Benjamin Kellenberger
'''

from typing import Iterable
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, execute_values
from util.helpers import LogDecorator
psycopg2.extras.register_uuid()


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
                with open(credentials, 'r', encoding='utf-8') as cred:
                    lines = cred.readlines()
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
        except Exception as exc:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception('Incomplete database credentials provided in configuration file ' + \
                f'(message: "{str(exc)}").') from exc

        try:
            self._create_connection_pool()
        except Exception as exc:
            if verbose_start:
                LogDecorator.print_status('fail')
            raise Exception(f'Could not connect to database (message: "{str(exc)}").') from exc

        if verbose_start:
            LogDecorator.print_status('ok')



    def _create_connection_pool(self):
        self.connection_pool = ThreadedConnectionPool(
            0,
            max(2, self.config.getProperty('Database', 'max_num_connections', type=int,
                    fallback=20)),  # 2 connections are needed as minimum for retrying of execution
            host=self.host,
            database=self.database,
            port=self.port,
            user=self.user,
            password=self.password,
            connect_timeout=10
        )



    def canConnect(self):
        with self.get_connection() as conn:
            return conn is not None and not conn.closed



    @contextmanager
    def get_connection(self):
        conn = self.connection_pool.getconn()
        conn.autocommit = True
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)       #, close=False)



    def execute(self, query, arguments, numReturn=None):
        if isinstance(arguments, Iterable) and len(arguments) == 0:
            arguments = None
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # execute statement
            try:
                cursor.execute(query, arguments)
                conn.commit()

                # get results
                return_values = []
                if numReturn is None:
                    return None

                if numReturn == 'all':
                    return_values = cursor.fetchall()
                    return return_values

                for _ in range(numReturn):
                    return_val = cursor.fetchone()
                    if return_val is None:
                        return return_values
                    return_values.append(return_val)
                return return_values

            except Exception as exc:
                print(exc)
                if not conn.closed:
                    conn.rollback()

            return None



    def execute_cursor(self, connection, query, arguments):
        if isinstance(arguments, Iterable) and len(arguments) == 0:
            arguments = None
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, arguments)
            connection.commit()

            return cursor

        except Exception as exc:
            print(exc)
            if not connection.closed:
                connection.rollback()



    def insert(self, query, values, numReturn=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                execute_values(cursor, query, values)
                conn.commit()

                # get results
                return_values = []
                if numReturn is None:
                    # cursor.close()
                    return None

                if numReturn == 'all':
                    return_values = cursor.fetchall()
                    return returnValues

                for _ in range(numReturn):
                    return_val = cursor.fetchone()
                    if return_val is None:
                        return return_values
                    return_values.append(return_val)
                return return_values

            except Exception as exc:
                print(exc)
                if not conn.closed:
                    conn.rollback()
