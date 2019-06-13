'''
    Provides functionality for checking login details,
    session validity, and the like.

    2019 Benjamin Kellenberger
'''

from modules.Database.app import Database
import pytz
from datetime import datetime
import secrets
import hashlib
import bcrypt
from .exceptions import *


class UserMiddleware():

    TOKEN_NUM_BYTES = 64
    SALT_NUM_ROUNDS = 24


    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)

        self.usersLoggedIn = {}    # username -> {timestamp, sessionToken}
    

    def _current_time(self):
        return datetime.now(tz=pytz.utc)


    def _create_token(self):
        return secrets.token_urlsafe(self.TOKEN_NUM_BYTES)


    def _compare_tokens(self, tokenA, tokenB):
        return secrets.compare_digest(tokenA, tokenB)


    def _check_password(self, providedPass, hashedTargetPass):
        return bcrypt.checkpw(providedPass, hashedTargetPass)
    

    def _create_hash(self, password):
        hash = bcrypt.hashpw(password, bcrypt.gensalt(self.SALT_NUM_ROUNDS))
        return hash


    def _init_or_extend_session(self, username, sessionToken=None):
        '''
            Establishes a "session" for the user (i.e., sets 'time_login'
            to now).
            Also creates a new sessionToken if None provided.
        '''
        now = self._current_time()

        if sessionToken is None:
            sessionToken = self._create_token()

        self.usersLoggedIn[username] = {
            'timestamp': now,
            'sessionToken': sessionToken
        }

        self.dbConnector.execute('''UPDATE {}.user SET time_login = %s, session_token = %s;
            WHERE username = %s
        '''.format(
            self.config.getProperty('Database', 'schema')
        ),
        (now, sessionToken, username,))
        #TODO: feedback that everything is ok?

        return sessionToken, now


    def _invalidate_session(self, username):
        del self.usersLoggedIn[username]
        self.dbConnector.execute(
            'UPDATE {}.user SET session_token = NULL WHERE username = %s'.format(
                self.config.getProperty('Database', 'schema')
            ),
            (username,))
        #TODO: feedback that everything is ok?


    def _check_account_exists(self, username):
        result = self.dbConnector.execute('SELECT name FROM {}.user WHERE name = %s'.format(
            self.config.getProperty('Database', 'schema')
        ),
        (username,))
        return len(result)>0


    def isLoggedIn(self, username, sessionToken):
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
        now = self._current_time()
        time_login = self.config.getProperty('UserHandler', 'time_login')
        if not username in self.usersLoggedIn:
            # check database
            sql = 'SELECT last_login, session_token FROM {}.user WHERE name = %s;'.format(
                self.config.getProperty('Database', 'schema')
            )
            result = self.dbConnector.execute(sql, username, numReturn=1)
            if len(result) == 0:
                # account does not exist
                self._invalidate_session(username)
                raise InvalidRequestException()


            # check for session token
            if not self._compare_tokens(result['session_token'], sessionToken):
                # invalid session token provided
                self._invalidate_session(username)
                raise InvalidRequestException()

            # check for timestamp
            if (now - result['last_login']) <= time_login:
                # user still logged in; extend session
                sessionToken, now = self._init_or_extend_session(username, sessionToken)
                return sessionToken, now

            else:
                # session time-out
                raise SessionTimeoutException()
            
            # generic error
            self._invalidate_session(username)
            raise InvalidRequestException()
        
        else:
            # check locally
            if not self._compare_tokens(self.usersLoggedIn[username]['sessionToken'],
                    sessionToken):
                # invalid session token provided
                self._invalidate_session(username)
                raise InvalidRequestException()

            if (now - self.usersLoggedIn[username]['timestamp']) <= time_login:
                # user still logged in; extend session
                self._init_or_extend_session(username, sessionToken)
                return True

            else:
                # session time-out
                raise SessionTimeoutException()

            # generic error
            self._invalidate_session(username)
            raise InvalidRequestException()
        
        # generic error
        self._invalidate_session(username)
        raise InvalidRequestException()



    def login(self, username, password):

        #TODO: check if logged in
        #TODO: SQL sanitize

        # get user info
        userData = self.dbConnector.execute(
            'SELECT hash FROM {}.user WHERE name = %s;'.format(
                self.config.getProperty('Database', 'schema')
            ),
            (username,)
        )
        if len(userData) == 0:
            # account does not exist
            raise InvalidRequestException()
        
        # verify provided password
        if self._check_password(password, userData['hash']):
            # correct
            sessionToken, timestamp = self._init_or_extend_session(username, None)
            return sessionToken, timestamp

        else:
            # incorrect
            self._invalidate_session(username)
            raise InvalidPasswordException()
    

    def accountExists(self, username):
        if self._check_account_exists(username):
            raise AccountExistsException(username)
        return False


    def createAccount(self, username, password, email):
        if not self.accountExists(username):
            hash = self._create_hash(password)
            sql = '''
                INSERT INTO {}.user (name, hash, email)
                VALUES (%s, %s, %s);
            '''.format(self.config.getProperty('Database', 'schema'))
            self.dbConnector.execute(sql,
            (username, hash, email))

            sessionToken, timestamp = self._init_or_extend_session(username)
            return sessionToken, timestamp
        