'''
    Provides functionality for checking login details,
    session validity, and the like.

    2019 Benjamin Kellenberger
'''

from threading import Thread
from modules.Database.app import Database
import psycopg2
from psycopg2 import sql
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
        result = self.dbConnector.execute('SELECT last_login, session_token FROM aide_admin.user WHERE name = %s;',
                                (username,), numReturn=1)
        if not len(result):
            return None
        result = result[0]
        return result


    def _extend_session_database(self, username, sessionToken):
        '''
            Updates the last login timestamp of the user to the current
            time and commits the changes to the database.
            Runs in a thread to be non-blocking.
        '''
        def _extend_session():
            now = self._current_time()

            self.dbConnector.execute('''UPDATE aide_admin.user SET last_login = %s,
                    session_token = %s
                    WHERE name = %s
                ''',
                (now, sessionToken, username,),
                numReturn=None)
            
            # also update local cache
            self.usersLoggedIn[username]['timestamp'] = now
        
        eT = Thread(target=_extend_session)
        eT.start()


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
            self.dbConnector.execute('''UPDATE aide_admin.user SET last_login = %s, session_token = %s
                WHERE name = %s
            ''',
            (now, sessionToken, username,),
            numReturn=None)
            
            # fetch user metadata and store locally
            userData = self._get_user_data(username)
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken
            }

        # update local cache as well
        if not username in self.usersLoggedIn:
            # fetch user metadata and store locally
            userData = self._get_user_data(username)
            
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken
            }
        else:
            self.usersLoggedIn[username]['timestamp'] = now
            self.usersLoggedIn[username]['sessionToken'] = sessionToken

            # also tell DB about updated tokens
            self._extend_session_database(username, sessionToken)

        expires = now + timedelta(0, self.config.getProperty('UserHandler', 'time_login', type=int))

        return sessionToken, now, expires


    def _invalidate_session(self, username):
        if username in self.usersLoggedIn:
            del self.usersLoggedIn[username]
        self.dbConnector.execute(
            'UPDATE aide_admin.user SET session_token = NULL WHERE name = %s',
            (username,),
            numReturn=None)
        #TODO: feedback that everything is ok?


    def _check_account_exists(self, username, email):
        response = {
            'username': True,
            'email': True
        }
        if username is None or not len(username): username = ''
        if email is None or not len(email): email = ''
        result = self.dbConnector.execute('SELECT COUNT(name) AS c FROM aide_admin.user WHERE name = %s UNION ALL SELECT COUNT(name) AS c FROM aide_admin.user WHERE email = %s',
                (username,email,),
                numReturn=2)

        response['username'] = (result[0]['c'] > 0)
        response['email'] = (result[1]['c'] > 0)

        return response


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
            time_diff = (now - result['last_login']).total_seconds()
            if time_diff <= time_login:
                # user still logged in
                if not username in self.usersLoggedIn:
                    self.usersLoggedIn[username] = {
                        'timestamp': now,
                        'sessionToken': sessionToken
                    }
                else:
                    self.usersLoggedIn[username]['timestamp'] = now

                # extend user session (commit to DB) if needed
                if time_diff >= 0.75 * time_login:
                    self._extend_session_database(username, sessionToken)

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
                # invalid session token provided; check database if token has updated
                # (can happen if user logs in again from another machine)
                result = self._get_user_data(username)
                if not self._compare_tokens(result['session_token'],
                            sessionToken):
                    return False
                
                else:
                    # update local cache
                    self.usersLoggedIn[username]['sessionToken'] = result['session_token']
                    self.usersLoggedIn[username]['timestamp'] = now

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


    def _check_authorized(self, project, username, admin):
        '''
            Verifies whether a user has access rights to a project.

            TODO: cache or not?
        '''
        if not project:
            return True

        if admin:
            queryStr = sql.SQL('SELECT COUNT(*) AS cnt FROM {} WHERE project = %s AND username = %s AND isAdmin = %s').format(
                sql.Identifier('aide_admin', 'authentication')
            )
            queryVals = (project,username,admin,)
        else:
            queryStr = sql.SQL('SELECT COUNT(*) AS cnt FROM {} WHERE project = %s AND username = %s').format(
                sql.Identifier('aide_admin', 'authentication')
            )
            queryVals = (project,username,)
        result = self.dbConnector.execute(queryStr, queryVals, 1)
        return result[0]['cnt'] == 1


    def _check_user_privileges(self, username, superuser=False, canCreateProjects=False):
        result = self.dbConnector.execute('''SELECT isSuperUser, canCreateProjects
            FROM aide_admin.user WHERE name = %s;''',
            (username,),
            1)
        if not len(result):
            return False
        if superuser and not result[0]['issuperuser']:
            return False
        if canCreateProjects and not (
            result[0]['cancreateprojects'] or result[0]['issuperuser']):
            return False
        return True


    def isAuthenticated(self, username, sessionToken, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False):
        '''
            Checks if the user is authenticated to access a service.
            Returns False if one or more of the following conditions holds:
            - user is not logged in
            - 'project' (shortname) is provided, project is configured to be private and user is not in the
                authenticated users list
            - 'admin' is True, 'project' (shortname) is provided and user is not an admin of the project
            - 'superuser' is True and user is not a super user
            - 'canCreateProjects' is True and user is not authenticated to create (or remove) projects

            If 'extend_session' is True, the user's session will automatically be prolonged by the max login time
            specified in the configuration file.
        '''
        if not self._check_logged_in(username, sessionToken):
            return False

        if not self._check_authorized(project, username, admin):
            return False

        if not self._check_user_privileges(username, superuser, canCreateProjects):
            return False
        
        if extend_session:
            self._init_or_extend_session(username, sessionToken)
        
        return True


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


    def getUserPermissions(self, project, username):
        '''
            Returns the user-to-project relation (e.g., if user is admin)
        '''
        queryStr = sql.SQL('SELECT * FROM {id_auth} WHERE project = %s AND username = %s').format(
            id_auth=sql.Identifier('aide_admin', 'authentication'))
        result = self.dbConnector.execute(queryStr, (project,username,), 1)[0]
        return {
            'isAdmin': result['isadmin']
        }


    def login(self, username, password, sessionToken):

        # check if logged in
        if self._check_logged_in(username, sessionToken):
            # still logged in; extend session
            sessionToken, now, expires = self._init_or_extend_session(username, sessionToken)
            return sessionToken, now, expires

        # get user info
        userData = self.dbConnector.execute(
            'SELECT hash FROM aide_admin.user WHERE name = %s;',
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


    def accountExists(self, username, email):
        return self._check_account_exists(username, email)


    def createAccount(self, username, password, email):
        accExstChck = self._check_account_exists(username, email)
        if accExstChck['username'] or accExstChck['email']:
            raise AccountExistsException(username)

        else:
            hash = self._create_hash(password.encode('utf8'))

            queryStr = '''
                INSERT INTO aide_admin.user (name, email, hash)
                VALUES (%s, %s, %s);
            '''
            self.dbConnector.execute(queryStr,
            (username, email, hash,),
            numReturn=None)
            sessionToken, timestamp, _, expires = self._init_or_extend_session(username)
            return sessionToken, timestamp, expires

    
    def getUserNames(self, project=None):
        if not project:
            queryStr = 'SELECT name FROM aide_admin.user'
            queryVals = None
        else:
            queryStr = 'SELECT username AS name FROM aide_admin.authentication WHERE project = %s'
            queryVals = (project,)
        result = self.dbConnector.execute(queryStr, queryVals, 'all')
        response = [r['name'] for r in result]
        return response