'''
    Provides functionality for checking login details,
    session validity, and the like.

    2019 Benjamin Kellenberger
'''

from modules.Database.app import Database
import psycopg2
from datetime import timedelta
from util.helpers import current_time
import secrets
import hashlib
import bcrypt
from .exceptions import *


class UserMiddleware():

    TOKEN_NUM_BYTES = 64
    SALT_NUM_ROUNDS = 12


    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self.usersLoggedIn = {}    # username -> {timestamp, sessionToken}
    

    def _current_time(self):
        return current_time()


    def _create_token(self):
        return secrets.token_urlsafe(self.TOKEN_NUM_BYTES)


    def _compare_tokens(self, tokenA, tokenB):
        if tokenA is None or tokenB is None:
            return False
        return secrets.compare_digest(tokenA, tokenB)


    def _check_password(self, providedPass, hashedTargetPass):
        return bcrypt.checkpw(providedPass, hashedTargetPass)
    

    def _create_hash(self, password):
        hash = bcrypt.hashpw(password, bcrypt.gensalt(self.SALT_NUM_ROUNDS))
        return hash


    def _get_user_data(self, username):
        sql = 'SELECT last_login, session_token, isAdmin FROM {}.user WHERE name = %s;'.format(
            self.config.getProperty('Database', 'schema')
        )
        result = self.dbConnector.execute(sql, (username,), numReturn=1)
        if not len(result):
            return None
        result = result[0]
        return result


    def _init_or_extend_session(self, username, sessionToken=None):
        '''
            Establishes a "session" for the user (i.e., sets 'time_login'
            to now).
            Also creates a new sessionToken if None provided.
        '''
        now = self._current_time()

        if sessionToken is None:
            sessionToken = self._create_token()

            # new session created; add to database
            self.dbConnector.execute('''UPDATE {}.user SET last_login = %s, session_token = %s
                WHERE name = %s
            '''.format(
                self.config.getProperty('Database', 'schema')
            ),
            (now, sessionToken, username,),
            numReturn=None)
            
            # fetch user metadata and store locally
            userData = self._get_user_data(username)
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken,
                'isAdmin': userData['isadmin']
            }


        # update local cache as well
        if not username in self.usersLoggedIn:
            # fetch user metadata and store locally
            userData = self._get_user_data(username)
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken,
                'isAdmin': userData['isadmin']
            }
        else:
            self.usersLoggedIn[username]['timestamp'] = now

        expires = now + timedelta(0, self.config.getProperty('UserHandler', 'time_login', type=int))

        return sessionToken, now, expires


    def _invalidate_session(self, username):
        if username in self.usersLoggedIn:
            del self.usersLoggedIn[username]
        self.dbConnector.execute(
            'UPDATE {}.user SET session_token = NULL WHERE name = %s'.format(
                self.config.getProperty('Database', 'schema')
            ),
            (username,),
            numReturn=None)
        #TODO: feedback that everything is ok?


    def _check_account_exists(self, username):
        result = self.dbConnector.execute('SELECT name FROM {}.user WHERE name = %s'.format(
            self.config.getProperty('Database', 'schema')
        ),
        (username,),
        numReturn=1)
        return len(result)>0


    def _check_logged_in(self, username, sessionToken):
        now = self._current_time()
        time_login = self.config.getProperty('UserHandler', 'time_login', type=int)
        if not username in self.usersLoggedIn:
            # check database
            result = self._get_user_data(username)
            if result is None:
                # account does not exist
                return False

            # check for session token
            if not self._compare_tokens(result['session_token'], sessionToken):
                # invalid session token provided
                return False

            # check for timestamp
            if (now - result['last_login']).total_seconds() <= time_login:
                # user still logged in
                if not username in self.usersLoggedIn:
                    self.usersLoggedIn[username] = {
                        'timestamp': now,
                        'sessionToken': sessionToken,
                        'isAdmin': result['isadmin']
                    }
                else:
                    self.usersLoggedIn[username]['timestamp'] = now
                return True

            else:
                # session time-out
                return False
            
            # generic error
            return False
        
        else:
            # check locally
            if not self._compare_tokens(self.usersLoggedIn[username]['sessionToken'],
                    sessionToken):
                # invalid session token provided
                return False

            if (now - self.usersLoggedIn[username]['timestamp']).total_seconds() <= time_login:
                # user still logged in
                return True

            else:
                # session time-out
                return False

            # generic error
            return False
        
        # generic error
        return False


    def isAuthenticated(self, username, sessionToken, admin=False):
        '''
            Checks if the user is logged in.
            If 'admin' is True, returns True only if the user is
            logged in and an administrator.
        '''
        loggedIn = self._check_logged_in(username, sessionToken)
        if not loggedIn:
            return False
        
        elif not admin:
            return True

        else:
            return username in self.usersLoggedIn and \
                'isAdmin' in self.usersLoggedIn[username] and \
                    self.usersLoggedIn[username]['isAdmin'] is True


    def getLoginData(self, username, sessionToken):
        '''
            Performs a lookup on the login timestamp dict.
            If the username cannot be found (also not in the database),
            they are not logged in (False returned).
            If the difference between the current time and the recorded
            login timestamp exceeds a pre-defined threshold, the user is
            removed from the dict and False is returned.
            Otherwise returns True if and only if 'sessionToken' matches
            the entry in the database.
        '''
        if self._check_logged_in(username, sessionToken):
            # still logged in; extend session
            sessionToken, now, expires = self._init_or_extend_session(username, sessionToken)
            return sessionToken, now, expires

        else:
            # not logged in or error
            raise Exception('Not logged in.')



    def login(self, username, password, sessionToken):
        #TODO: SQL sanitize

        # check if logged in
        try:
            sessionToken, timestamp, expires = self._check_logged_in(username, sessionToken)
            return sessionToken, timestamp, expires

        except:
            # not logged in; continue
            pass

        # get user info
        userData = self.dbConnector.execute(
            'SELECT hash FROM {}.user WHERE name = %s;'.format(
                self.config.getProperty('Database', 'schema')
            ),
            (username,),
            numReturn=1
        )
        if len(userData) == 0:
            # account does not exist
            raise InvalidRequestException()
        userData = userData[0]
        
        # verify provided password
        if self._check_password(password.encode('utf8'), bytes(userData['hash'])):
            # correct
            sessionToken, timestamp, expires = self._init_or_extend_session(username, None)
            return sessionToken, timestamp, expires

        else:
            # incorrect
            self._invalidate_session(username)
            raise InvalidPasswordException()
    

    def logout(self, username, sessionToken):
        # check if logged in first
        if self._check_logged_in(username, sessionToken):
            self._invalidate_session(username)


    def accountExists(self, username):
        return self._check_account_exists(username)


    def createAccount(self, username, password, email):
        if self._check_account_exists(username):
            raise AccountExistsException(username)

        else:
            hash = self._create_hash(password.encode('utf8'))

            sql = '''
                INSERT INTO {}.user (name, email, hash)
                VALUES (%s, %s, %s);
            '''.format(self.config.getProperty('Database', 'schema'))
            self.dbConnector.execute(sql,
            (username, email, hash,),
            numReturn=None)
            sessionToken, timestamp, expires = self._init_or_extend_session(username)
            return sessionToken, timestamp, expires