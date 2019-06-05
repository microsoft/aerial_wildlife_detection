'''
    Helper functions for database access.

    2019 Benjamin Kellenberger
'''

import psycopg2
from util.singleton import Singleton



@Singleton
class DBConnector:

    def init(self, config):

        # get DB parameters
        self.host = config['DATABASE']['host']
        self.port = config['DATABASE']['port']
        self.user = config['DATABASE']['user']
        self.password = config['DATABASE']['password']


    def _createConnection(self):
        try:
            self.conn = psycopg2.connect(database=self.host,
                                        user=self.user,
                                        password=self.password)
        except:
            print('Error connecting to database {}:{} with username {}.'.format(
                self.host, self.port, self.user
            ))
            #TODO: next steps