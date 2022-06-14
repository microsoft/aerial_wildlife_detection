'''
    Provides functionality for checking login details,
    session validity, and the like.

    2019-22 Benjamin Kellenberger
'''

import os
import functools
import hashlib
from threading import Thread
from psycopg2 import sql
from datetime import timedelta
from util.helpers import current_time, checkDemoMode
import secrets
import bcrypt
from bottle import request, response, abort
from .exceptions import *


class UserMiddleware():

    TOKEN_NUM_BYTES = 64
    SALT_NUM_ROUNDS = 12

    CSRF_TOKEN_NAME = '_csrf_token'
    CSRF_TOKEN_AGE = 600
    CSRF_SECRET_FALLBACK = '!ftU$_4FnJ6eA2uN'   # fallback for CSRF secret; used if not defined in config


    def __init__(self, config, dbConnector):
        self.config = config
        self.dbConnector = dbConnector      #Database(config)

        self.usersLoggedIn = {}    # username -> {timestamp, sessionToken}
    
        self.csrfSecret = self.config.getProperty('UserHandler', 'csrf_secret', type=str, fallback=self.CSRF_SECRET_FALLBACK)


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
    

    @classmethod
    def _create_hash(cls, password):
        hash = bcrypt.hashpw(password, bcrypt.gensalt(cls.SALT_NUM_ROUNDS))
        return hash


    def _get_user_data(self, username):
        result = self.dbConnector.execute('SELECT last_login, session_token, secret_token FROM aide_admin.user WHERE name = %s;',
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
            
            # store locally
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken
            }

        # update local cache as well
        if not username in self.usersLoggedIn:
            self.usersLoggedIn[username] = {
                'timestamp': now,
                'sessionToken': sessionToken
            }
        else:
            self.usersLoggedIn[username]['timestamp'] = now
            self.usersLoggedIn[username]['sessionToken'] = sessionToken

            # also tell DB about updated tokens
            self._extend_session_database(username, sessionToken)

        time_login = self.config.getProperty('UserHandler', 'time_login', type=int, fallback=31536000)  # fallback: add one year
        expires = now + timedelta(0, time_login)

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
        time_login = self.config.getProperty('UserHandler', 'time_login', type=int, fallback=-1)
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
            if time_login <= 0 or time_diff <= time_login:
                # user still logged in
                if not username in self.usersLoggedIn:
                    self.usersLoggedIn[username] = {
                        'timestamp': now,
                        'sessionToken': sessionToken
                    }
                else:
                    self.usersLoggedIn[username]['timestamp'] = now

                # extend user session (commit to DB) if needed
                if time_login > 0 and time_diff >= 0.75 * time_login:
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

            time_diff = (now - self.usersLoggedIn[username]['timestamp']).total_seconds()
            if time_login <= 0 or time_diff <= time_login:
                # user still logged in; extend user session (commit to DB) if needed
                if time_login > 0 and time_diff >= 0.75 * time_login:
                    self._extend_session_database(username, sessionToken)
                return True

            else:
                # local cache session time-out; check if database holds more recent timestamp
                result = self._get_user_data(username)
                if time_login <= 0 or (now - result['last_login']).total_seconds() <= time_login:
                    # user still logged in; update
                    self._init_or_extend_session(username, sessionToken)

                else:
                    # session time-out
                    return False

            # generic error
            return False
        
        # generic error
        return False


    def _check_authorized(self, project, username, admin, return_all=False):
        '''
            Verifies whether a user has access rights to a project.
            If "return_all" is set to True, a dict with the following bools
            is returned:
            - enrolled: if the user is member of the project
            - isAdmin: if the user is a project administrator
            - isPublic: if the project is publicly visible (*)
            - demoMode: if the project runs in demo mode (*)

            (* note that these are here for convenience, but do not count
            as authorization tokens)


            If "return_all" is False, only a single bool is returned, with
            criteria as follows:
            - if "admin" is set to True, the user must be a project admini-
              strator
            - else, the user must be enrolled, admitted, and not blocked for
              the current date and time

            In this case, options like the demo mode and public flag are not
            relevant for the decision.
        '''
        now = current_time()
        response = {
            'enrolled': False,
            'isAdmin': False,
            'isPublic': False
        }

        queryStr = sql.SQL('''
            SELECT * FROM aide_admin.authentication AS auth
            JOIN (SELECT shortname, demoMode, isPublic FROM aide_admin.project) AS proj
            ON auth.project = proj.shortname
            WHERE project = %s AND username = %s;
        ''')
        try:
            result = self.dbConnector.execute(queryStr, (project, username,), 1)
            if len(result):
                response['isAdmin'] = result[0]['isadmin']
                response['isPublic'] = result[0]['ispublic']
                admitted_until = True
                blocked_until = False
                if result[0]['admitted_until'] is not None:
                    admitted_until = (result[0]['admitted_until'] >= now)
                if result[0]['blocked_until'] is not None:
                    blocked_until = (result[0]['blocked_until'] >= now)
                response['enrolled'] = (admitted_until and not blocked_until)
        except:
            # no results to fetch: user is not authenticated
            pass
    
        # check if super user
        superUser = self._check_user_privileges(username, superuser=True)
        if superUser:
            response['enrolled'] = True
            response['isAdmin'] = True

        if return_all:
            return response
        else:
            if admin:
                return response['isAdmin']
            else:
                return response['enrolled']
            
        # if admin:
        #     queryStr = sql.SQL('''SELECT COUNT(*) AS cnt FROM aide_admin.authentication
        #         WHERE project = %s AND username = %s AND isAdmin = %s''')
        #     queryVals = (project,username,admin,)
        # else:
        #     queryStr = sql.SQL('''SELECT COUNT(*) AS cnt FROM aide_admin.authentication
        #         WHERE project = %s AND username = %s
        #         AND (
        #             (admitted_until IS NULL OR admitted_until >= now())
        #             AND
        #             (blocked_until IS NULL OR blocked_until < now())
        #         )''')
        #     queryVals = (project,username,)
        # result = self.dbConnector.execute(queryStr, queryVals, 1)
        # return result[0]['cnt'] == 1


    def checkDemoMode(self, project):
        return checkDemoMode(project, self.dbConnector)


    def decryptSessionToken(self, username, request):
        try:
            userdata = self._get_user_data(username)
            return request.get_cookie('session_token', secret=userdata['secret_token'])
        except:
            return None


    def encryptSessionToken(self, username, response):
        userdata = self._get_user_data(username)
        response.set_cookie('session_token', userdata['session_token'],
                            httponly=True, path='/', secret=userdata['secret_token'])


    def _check_user_privileges(self, username, superuser=False, canCreateProjects=False, return_all=False):
        response = {
            'superuser': False,
            'can_create_projects': False
        }
        result = self.dbConnector.execute('''SELECT isSuperUser, canCreateProjects
            FROM aide_admin.user WHERE name = %s;''',
            (username,),
            1)
        
        if len(result):
            response['superuser'] = result[0]['issuperuser']
            response['can_create_projects'] = result[0]['cancreateprojects']

        if return_all:
            return response
        
        else:
            if superuser and not result[0]['issuperuser']:
                return False
            if canCreateProjects and not (
                result[0]['cancreateprojects'] or result[0]['issuperuser']):
                return False
            return True


    def isAuthenticated(self, username, sessionToken, project=None, admin=False, superuser=False, canCreateProjects=False, extend_session=False, return_all=False):
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
            If 'return_all' is True, all individual flags (instead of just a single bool) is returned.
        '''

        demoMode = checkDemoMode(project, self.dbConnector)

        if return_all:
            returnVals = {}
            returnVals['logged_in'] = self._check_logged_in(username, sessionToken)
            if not returnVals['logged_in']:
                username = None
            if project is not None:
                returnVals['project'] = self._check_authorized(project, username, admin, return_all=True)
                returnVals['project']['demoMode'] = demoMode
            returnVals['privileges'] = self._check_user_privileges(username, superuser, canCreateProjects, return_all=True)
            if returnVals['logged_in'] and extend_session:
                self._init_or_extend_session(username, sessionToken)
            return returnVals

        else:
            # return True if project is in demo mode
            if demoMode is not None and demoMode:
                return True
            if not self._check_logged_in(username, sessionToken):
                return False
            if project is not None and not self._check_authorized(project, username, admin):
                return False
            if not self._check_user_privileges(username, superuser, canCreateProjects):
                return False
            if extend_session:
                self._init_or_extend_session(username, sessionToken)
            return True


    def getAuthentication(self, username, project=None):
        '''
            Returns general authentication properties of the user, regardless of whether
            they are logged in or not.
            If a project shortname is specified, this will also return the user access
            properties for the given project.
        '''
        
        response = {}
        if project is None:
            result = self.dbConnector.execute(
                '''SELECT * FROM aide_admin.user AS u
                    WHERE name = %s;
                ''',
                (username,),
                1)
            response['canCreateProjects'] = result[0]['cancreateprojects']
            response['isSuperUser'] = result[0]['issuperuser']
        else:
            result = self.dbConnector.execute(
                '''SELECT * FROM aide_admin.user AS u
                    JOIN aide_admin.authentication AS a
                    ON u.name = a.username
                    WHERE name = %s
                    AND project = %s;
                ''',
                (username,project,),
                1)
            response['canCreateProjects'] = result[0]['cancreateprojects']
            response['isSuperUser'] = result[0]['issuperuser']
            response['isAdmin'] = result[0]['isadmin']
            response['admittedUntil'] = result[0]['admitted_until']
            response['blockedUntil'] = result[0]['blocked_until']

        return response


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
            Returns the user-to-project relation (e.g., if user is admin).
        '''
        response = {
            'demoMode': False,
            'isAdmin': False,
            'admittedUntil': None,
            'blockedUntil': None
        }

        try:
            # demo mode
            response['demoMode'] = checkDemoMode(project, self.dbConnector)

            # rest
            queryStr = sql.SQL('SELECT * FROM {id_auth} WHERE project = %s AND username = %s').format(
                id_auth=sql.Identifier('aide_admin', 'authentication'))
            result = self.dbConnector.execute(queryStr, (project,username,), 1)
            if len(result):
                response['isAdmin'] = result[0]['isadmin']
                response['admittedUntil'] = result[0]['admitted_until']
                response['blockedUntil'] = result[0]['blocked_until']

        finally:
            return response


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
            sessionToken, timestamp, expires = self._init_or_extend_session(username)
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


    def setPassword(self, username, password):
        hashVal = self._create_hash(password.encode('utf8'))
        queryStr = '''
            UPDATE aide_admin.user
            SET hash = %s
            WHERE name = %s;
            SELECT hash
            FROM aide_admin.user
            WHERE name = %s;
        '''
        result = self.dbConnector.execute(queryStr, (hashVal, username, username), 1)
        if len(result):
            return {
                'success': True
            }
        else:
            return {
                'success': False,
                'message': f'User with name "{username}" does not exist.'
            }
    

    # CSRF prevention functionality
    # adapted from https://github.com/Othernet-Project/bottle-utils-csrf/blob/master/bottle_utils/csrf.py

    def generate_csrf_token(self):
        '''
            Generate and set new CSRF token in cookie. The generated token is set to
            ``request.csrf_token`` attribute for easier access by other functions.

            .. warning::

            This function uses ``os.urandom()`` call to obtain 8 random bytes when
            generating the token. It is possible to deplete the randomness pool and
            make the random token predictable.
        '''
        sha256 = hashlib.sha256()
        sha256.update(os.urandom(8))
        token = sha256.hexdigest().encode('utf8')
        response.set_cookie(self.CSRF_TOKEN_NAME, token, path='/',
                            secret=self.csrfSecret, max_age=self.CSRF_TOKEN_AGE)
        request.csrf_token = token.decode('utf8')
    

    def csrf_token(self, func):
        '''
            Create and set CSRF token in preparation for subsequent POST request. This
            decorator is used to set the token. It also sets the ``'Cache-Control'``
            header in order to prevent caching of the page on which the token appears.
            When an existing token cookie is found, it is reused. The existing token is
            reset so that the expiration time is extended each time it is reused.
            The POST handler must use the :py:func:`~bottle_utils.csrf.csrf_protect`
            decotrator for the token to be used in any way.
            The token is available in the ``bottle.request`` object as ``csrf_token``
            attribute::
                @app.get('/')
                @bottle.view('myform')
                @csrf_token
                def put_token_in_form():
                    return dict(token=request.csrf_token)
            In a view, you can render this token as a hidden field inside the form. The
            hidden field must have a name ``_csrf_token``::
                <form method="POST">
                    <input type="hidden" name="_csrf_token" value="{{ csrf_token }}">
                    ....
                </form>
        '''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            token = request.get_cookie(self.CSRF_TOKEN_NAME, secret=self.csrfSecret)
            if token:
                # We will reuse existing tokens
                response.set_cookie(self.CSRF_TOKEN_NAME, token, path='/',
                                    secret=self.csrfSecret, max_age=self.CSRF_TOKEN_AGE)
                request.csrf_token = token.decode('utf8')
            else:
                self.generate_csrf_token()
            # Pages with CSRF tokens should not be cached
            response.headers[str('Cache-Control')] = ('no-cache, max-age=0, '
                                                    'must-revalidate, no-store')
            return func(*args, **kwargs)
        return wrapper


    def csrf_check(self, token, regenerate=True):
        '''
            Perform CSRF protection checks. Performs checks to determine if
            submitted token matches the token in the cookie.
            If "regenerate" is False the current CSRF token will be reused.
        '''
        if isinstance(token, bytes):
            token = token.decode('utf8')
        token_cookie = request.get_cookie(self.CSRF_TOKEN_NAME, secret=self.csrfSecret)
        if isinstance(token_cookie, bytes):
            token_cookie = token_cookie.decode('utf8')
        if token != token_cookie:
            response.delete_cookie(self.CSRF_TOKEN_NAME, path='/', secret=self.csrfSecret,
                                max_age=self.CSRF_TOKEN_AGE)
            abort(403, 'Request is invalid or has expired.')
        if regenerate:
            self.generate_csrf_token()