'''
    Handles data flow about projects in general.

    2019-20 Benjamin Kellenberger
'''

from psycopg2 import sql
from modules.Database.app import Database
from util.helpers import current_time


class ReceptionMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)


    def get_project_info(self, username=None, isSuperUser=False):
        '''
            Returns metadata about projects:
            - names
            - whether the projects are archived or not
            - links to interface (if user is authenticated)
            - requests for authentication (else)    TODO
            - links to stats and review page (if admin) TODO
            - etc.
        '''
        now = current_time()

        if isSuperUser:
            authStr = sql.SQL('')
            queryVals = None
        elif username is not None:
            authStr = sql.SQL('WHERE username = %s OR demoMode = TRUE OR isPublic = TRUE')
            queryVals = (username,)
        else:
            authStr = sql.SQL('WHERE demoMode = TRUE OR isPublic = TRUE')
            queryVals = None
        
        queryStr = sql.SQL('''SELECT shortname, name, description, archived,
            username, isAdmin,
            admitted_until, blocked_until,
            annotationType, predictionType, isPublic, demoMode, interface_enabled, archived, ai_model_enabled,
            CASE WHEN username = owner THEN TRUE ELSE FALSE END AS is_owner
            FROM aide_admin.project AS proj
            FULL OUTER JOIN (SELECT * FROM aide_admin.authentication
            ) AS auth ON proj.shortname = auth.project
            {authStr};
        ''').format(authStr=authStr)

        result = self.dbConnector.execute(queryStr, queryVals, 'all')
        response = {}
        if result is not None and len(result):
            for r in result:
                projShort = r['shortname']
                if not projShort in response:
                    userAdmitted = True
                    if r['admitted_until'] is not None and r['admitted_until'] < now:
                        userAdmitted = False
                    if r['blocked_until'] is not None and r['blocked_until'] >= now:
                        userAdmitted = False
                    response[projShort] = {
                        'name': r['name'],
                        'description': r['description'],
                        'archived': r['archived'],
                        'isOwner': r['is_owner'],
                        'annotationType': r['annotationtype'],
                        'predictionType': r['predictiontype'],
                        'isPublic': r['ispublic'],
                        'demoMode': r['demomode'],
                        'interface_enabled': r['interface_enabled'] and not r['archived'],
                        'aiModelEnabled': r['ai_model_enabled'],
                        'userAdmitted': userAdmitted
                    }
                if isSuperUser:
                    response[projShort]['role'] = 'super user'
                elif username is not None and r['username'] == username:
                    if r['isadmin']:
                        response[projShort]['role'] = 'admin'
                    else:
                        response[projShort]['role'] = 'member'
        
        return response

    
    def enroll_in_project(self, project, username, secretToken=None):
        '''
            Adds the user to the project if it allows arbitrary
            users to join. Returns True if this succeeded, else
            False.
        '''
        try:
            # check if project is public, and whether user is already member of it
            queryStr = sql.SQL('''SELECT isPublic, secret_token
            FROM aide_admin.project
            WHERE shortname = %s;
            ''')
            result = self.dbConnector.execute(queryStr, (project,), 1)

            # only allow enrolment if project is public, or else if secret tokens match
            if not len(result):
                return False
            elif not result[0]['ispublic']:
                # check if secret tokens match
                if secretToken is None or secretToken != result[0]['secret_token']:
                    return False
            
            # add user
            queryStr = '''INSERT INTO aide_admin.authentication (username, project, isAdmin)
            VALUES (%s, %s, FALSE)
            ON CONFLICT (username, project) DO NOTHING;
            '''
            self.dbConnector.execute(queryStr, (username,project,), None)
            return True
        except Exception as e:
            print(e)
            return False

    
    def getSampleImages(self, project, limit=128):
        '''
            Returns sample image URLs from the specified project for backdrop
            visualization on the landing page.
            Images are sorted descending according to the following criteria,
            in a row:
            1. last_requested
            2. date_added
            3. number of annotations
            4. number of predictions
            5. random
        '''
        queryStr = sql.SQL('''
            SELECT filename FROM {id_img} AS img
            LEFT OUTER JOIN (
                SELECT img_anno AS img_id, cnt_anno, cnt_pred FROM (
                    SELECT image AS img_anno, COUNT(image) AS cnt_anno
                    FROM {id_anno}
                    GROUP BY img_anno
                ) AS anno
                FULL OUTER JOIN (
                    SELECT image AS img_pred, COUNT(image) AS cnt_pred
                    FROM {id_pred}
                    GROUP BY img_pred
                ) AS pred
                ON anno.img_anno = pred.img_pred
            ) AS metaQuery
            ON img.id = metaQuery.img_id
            ORDER BY last_requested DESC NULLS LAST, date_added DESC NULLS LAST,
                cnt_anno DESC NULLS LAST, cnt_pred DESC NULLS LAST, random()
            LIMIT %s;
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_anno=sql.Identifier(project, 'annotation'),
            id_pred=sql.Identifier(project, 'prediction')
        )
        result = self.dbConnector.execute(queryStr, (limit,), 'all')
        response = []
        for r in result:
            response.append(r['filename'])
        return response