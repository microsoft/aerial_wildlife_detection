'''
    Helper function that imports MS-COCO labels into the database.
    Needs to be run on the file server.
    This assumes that the images of the dataset have already been placed in the
    folder of the connected file server (parameter "staticfiles_dir" in section [FileServer]
    of .ini file), and that the labels provided here match with a simple file directory lookup.
    Also pushes images to the database if they are not already in there.

    Inputs:
    - label-files: str or iterable of str for n JSON files holding annotation definitions

    Labels are stored in the "ANNOTATIONS" table, with the following non-standard field
    values:
        - timeCreated: (timestamp of the launch of this script)
        - timeRequired: -1
    Also adds class definitions.

    2019-22 Benjamin Kellenberger
'''

import os
import argparse
from psycopg2 import sql
from util import drivers


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse YOLO annotations and import into database.')
    parser.add_argument('--project', type=str,
                    help='Project shortname for which to export annotations.')
    parser.add_argument('--username', type=str,
                    help='Username under which the annotations should be registered. If not provided, the first administrator name in alphabetic order will be used.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--label-files', type=str, nargs='+',
                    help='One or more paths to COCO JSON files containing annotations.')
    
    args = parser.parse_args()
    

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)
    
    import json
    import glob
    from collections import defaultdict
    from tqdm import tqdm
    import datetime
    from PIL import Image
    from util.configDef import Config
    from modules import Database
    drivers.init_drivers()

    currentDT = datetime.datetime.now()
    currentDT = '{}-{}-{} {}:{}:{}'.format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)

    config = Config()
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')

    # check if running on file server
    imgBaseDir = config.getProperty('FileServer', 'staticfiles_dir')
    if not os.path.isdir(imgBaseDir):
        raise Exception('"{}" is not a valid directory on this machine. Are you running the script from the file server?'.format(imgBaseDir))

    # parse all files
    for f in args.label_files:
        classes = {}
        images = {}
        labels = defaultdict(list)

        meta = json.load(open(f, 'r'))
        for cat in meta['categories']:
            classes[cat['id']] = cat['name']
        for img in meta['images']:
            images[img['id']] = [img['file_name'], img['width'], img['height']]
        for anno in meta['annotations']:
            bbox = anno['bbox']
            label = anno['category_id']
            imgID = anno['image_id']

            # convert to rel format
            bbox[0] += bbox[2]/2.0
            bbox[1] += bbox[3]/2.0
            bbox[0] /= images[imgID][1]
            bbox[1] /= images[imgID][2]
            bbox[2] /= images[imgID][1]
            bbox[3] /= images[imgID][2]
            labels[imgID].append([label, bbox])

        # push classes to database
        classdef = {}
        for clID in classes.keys():
            dbConn.execute(sql.SQL('''
                INSERT INTO {id_lc} (name)
                VALUES (%s)
                ON CONFLICT(name) DO NOTHING;
            ''').format(id_lc=sql.Identifier(args.project, 'labelclass')), (classes[clID],))

            # get newly assigned UUID
            returnVal = dbConn.execute(sql.SQL('''
                SELECT id FROM {id_lc} WHERE name LIKE %s''').format(
                    id_lc=sql.Identifier(args.project, 'labelclass')),
            (classes[clID]+'%',),
            1)
            classdef[clID] = returnVal[0]['id']

        # get username
        usernames = dbConn.execute('''
            SELECT username FROM aide_admin.authentication
            WHERE project = %s
            AND isAdmin = TRUE
            ORDER BY username ASC
            ''',
            (args.project,),
            'all'
        )
        usernames = [u['username'] for u in usernames]
        username = usernames[0]
        if args.username is not None:
            if args.username not in usernames:
                print(f'WARNING: username "{args.username}" not found, using "{username}" instead.')
            else:
                username = args.username
        
        # insert annotations
        print(f'Inserting annotations under username "{username}".')

        queryStr = sql.SQL('''
            INSERT INTO {id_anno} (username, image, timeCreated, timeRequired, label, x, y, width, height)
            VALUES(
                %s,
                (SELECT id FROM {id_img} WHERE filename = %s),
                (TIMESTAMP %s),
                -1,
                %s,
                %s,
                %s,
                %s,
                %s
            );''').format(
                id_anno=sql.Identifier(args.project, 'annotation'),
                id_img=sql.Identifier(args.project, 'image')
            )

        # import
        for imgID in tqdm(labels.keys()):
            imgName = images[imgID][0]
            for anno in labels[imgID]:
                label = classdef[anno[0]]
                bbox = anno[1]
                dbConn.execute(queryStr,
                    (username, imgName, currentDT, label, bbox[0], bbox[1], bbox[2], bbox[3]))