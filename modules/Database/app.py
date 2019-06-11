'''
    Database connection functionality.

    2019 Benjamin Kellenberger
'''

import psycopg2


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
        except:
            print('Error connecting to database {}:{} with username {}.'.format(
                self.host, self.port, self.user
            ))
            #TODO: next steps


    def runServer(self):
        ''' Dummy function for compatibility reasons '''
        return

    
    def execute(self, sql, arguments, numReturn=None):
        cursor = self.conn.cursor()
        cursor.execute(sql, arguments)
        self.conn.commit()

        if numReturn is None:
            cursor.close()
            return

        returnValues = []
        for n in range(numReturn):
            rv = cursor.fetchone()
            if rv is None:
                return returnValues
            returnValues.append(rv)
 
        cursor.close()
        return returnValues