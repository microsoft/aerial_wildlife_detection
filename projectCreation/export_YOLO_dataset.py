'''
    Pull annotations and predictions from the database and export them into a
    folder according to the YOLO standard.

    2019-20 Benjamin Kellenberger
'''

import os
import argparse
from psycopg2 import sql

def _replace(string, oldTokens, newToken):
    if isinstance(oldTokens, str):
        oldTokens = [oldTokens]
    for o in oldTokens:
        string = string.replace(o, newToken)
    return string


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export annotations from database to YOLO format.')
    parser.add_argument('--project', type=str,
                    help='Project shortname for which to export annotations.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--target_folder', type=str, default='export', const=1, nargs='?',
                    help='Export directory for the annotation text files.')
    parser.add_argument('--export_annotations', type=bool, default=True, const=1, nargs='?',
                    help='Whether to export annotations (default: True).')
    parser.add_argument('--limit_users', type=str,
                    help='Which users (comma-separated list of usernames) to limit annotations to (default: None).')
    parser.add_argument('--exclude_users', type=str, default=None, const=1, nargs='?',
                    help='Comma-separated list of usernames whose annotations not to include (default: None).')
    #TODO: implement:
    # parser.add_argument('--export_predictions', type=bool, default=False, const=1, nargs='?',
    #                 help='Whether to export predictions (default: False).')
    # parser.add_argument('--append_confidences', type=bool, default=False, const=1, nargs='?',
    #                 help='Whether to append confidences to each prediction (default: False).')
    # parser.add_argument('--predictions_min_date', type=str, default=None, const=1, nargs='?',
    #                 help='Timestamp of earliest predictions to consider (default: None, i.e. all).')
    args = parser.parse_args()


    # setup
    print('Setup...\n')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    from tqdm import tqdm
    from util.configDef import Config
    from modules import Database

    config = Config()

    # setup DB connection
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')
    dbSchema = config.getProperty('Database', 'schema')

    # check if correct type of annotations
    annotationType = dbConn.execute('SELECT annotationtype FROM aide_admin.project WHERE shortname = %s;',
                            (args.project,), 1)
    if not len(annotationType):
        raise Exception(f'Project with name "{args.project}" could not be found in database.')
    annotationType = annotationType[0]['annotationtype']
    exportAnnotations = args.export_annotations
    if exportAnnotations and not (annotationType == 'boundingBoxes'):
        print('Warning: project annotations are not bounding boxes; skipping annotation export...')
        exportAnnotations = False


    # query label definition
    labeldef = {}   # label UUID : (name, index,)
    labeldef_inv = []   # list of label names in order (for export)
    labelQuery = dbConn.execute(sql.SQL('SELECT * FROM {id_lc};').format(id_lc=sql.Identifier(args.project, 'labelclass')), None, 'all')
    for labelIdx, l in enumerate(labelQuery):
        labelID = l['id']
        labelName = l['name']
        labeldef[labelID] = (labelName, labelIdx,)
        labeldef_inv.append(labelName)


    # start querying
    output = {}     # image name : YOLO text file contents

    if exportAnnotations:

        queryArgs = []
        
        # included and excluded users
        if args.limit_users is not None:
            limitUsers = []
            for u in args.limit_users.split(','):
                limitUsers.append(u.strip())
            sql_limitUsers = sql.SQL('WHERE anno.username IN %s')
            queryArgs = []
            queryArgs.append(tuple(limitUsers))
        else:
            sql_limitUsers = sql.SQL('')

        if args.exclude_users is not None:
            excludeUsers = []
            for u in args.exclude_users.split(','):
                excludeUsers.append(u.strip())
            if args.limit_users is not None:
                sql_excludeUsers = sql.SQL('AND anno.username NOT in %s')
            else:
                sql_excludeUsers = sql.SQL('WHERE anno.username IN %s')
            queryArgs.append(tuple(excludeUsers))
        else:
            sql_excludeUsers = sql.SQL('')

        if len(queryArgs) == 0:
            queryArgs = None

        queryStr = sql.SQL('''
            SELECT * FROM {id_anno} AS anno
            JOIN (SELECT id AS imID, filename FROM {id_img}) AS img
            ON anno.image = img.imID
            {sql_limitUsers}
            {sql_excludeUsers}
        ''').format(
            id_anno=sql.Identifier(args.project, 'annotation'),
            id_img=sql.Identifier(args.project, 'image'),
            sql_limitUsers=sql_limitUsers,
            sql_excludeUsers=sql_excludeUsers
        )
        
        cursor = dbConn.execute_cursor(queryStr, queryArgs)


        # iterate
        print('Querying database...\n')
        while True:
            nextItem = cursor.fetchone()
            if nextItem is None:
                break
            
            # parse
            if nextItem['label'] is None:
                # TODO: it might happen that an annotation has no label; skip in this case
                continue
                
            imgName = nextItem['filename']
            label = labeldef[nextItem['label']][1]      # store label index
            x = nextItem['x']
            y = nextItem['y']
            w = nextItem['width']
            h = nextItem['height']

            # append
            if not imgName in output:
                output[imgName] = []
            output[imgName].append('{} {} {} {} {}\n'.format(str(label), x, y, w, h))
        

    # write to disk
    if len(output) > 0:
        print('Writing data to disk...\n')
        
        # write label definition file
        os.makedirs(args.target_folder, exist_ok=True)
        labeldefPath = os.path.join(args.target_folder, 'classes.txt')
        with open(labeldefPath, 'w') as f:
            for l in labeldef_inv:
                f.write('{}\n'.format(l))

        for key in tqdm(output.keys()):

            labelFileName = _replace(key, ['.JPG','.jpg','.JPEG','.jpeg','.PNG','.png','.GIF','.gif','.NEF','.nef'], '.txt')
            targetFile = os.path.join(args.target_folder, labelFileName)
            targetFolder = os.path.split(targetFile)[0]
            os.makedirs(targetFolder, exist_ok=True)

            with open(targetFile, 'w') as f:
                for line in output[key]:
                    f.write(line)