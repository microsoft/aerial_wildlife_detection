'''
    2019-21 Benjamin Kellenberger
'''

import os
import argparse
from psycopg2 import sql
from util import drivers


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse YOLO annotations and import into database.')
    parser.add_argument('--project', type=str,
                    help='Project shortname for which to export annotations.')
    parser.add_argument('--username', type=str, default=None, const=1, nargs='?',
                    help='Username under which the annotations should be registered. If not provided, the first administrator name in alphabetic order will be used.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--label_folder', type=str,
                    help='Directory (absolute path) on this machine that contains the YOLO label text files.')
    parser.add_argument('--annotation_type', type=str, default='annotation', const=1, nargs='?',
                    help='Kind of the provided annotations. One of {"annotation", "prediction"} (default: annotation)')
    args = parser.parse_args()

    
    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)
    
    import glob
    from tqdm import tqdm
    import datetime
    import numpy as np
    from PIL import Image
    import base64
    from util.configDef import Config
    from modules import Database
    drivers.init_drivers()

    if args.label_folder == '':
        args.label_folder = None

    if args.label_folder is not None and not args.label_folder.endswith(os.sep):
        args.label_folder += os.sep

    currentDT = datetime.datetime.now()
    currentDT = '{}-{}-{} {}:{}:{}'.format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)

    config = Config()
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')


    # check if running on file server
    imgBaseDir = os.path.join(config.getProperty('FileServer', 'staticfiles_dir'), args.project)
    if not os.path.isdir(imgBaseDir):
        raise Exception('"{}" is not a valid directory on this machine. Are you running the script from the file server?'.format(imgBaseDir))

    if args.label_folder is not None and not os.path.isdir(args.label_folder):
        raise Exception('"{}" is not a valid directory on this machine.'.format(args.label_folder))

    if not imgBaseDir.endswith(os.sep):
        imgBaseDir += os.sep


    # parse class names and indices
    if args.label_folder is not None:
        with open(os.path.join(args.label_folder, 'classes.txt'),'r') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            className = line.strip()            

            # push to database
            dbConn.execute(sql.SQL('''
                INSERT INTO {id_lc} (name, idx)
                VALUES (
                    %s, %s
                )
                ON CONFLICT (name) DO NOTHING;
            ''').format(id_lc=sql.Identifier(args.project, 'labelclass')),
            (className,idx,))

        # prepare insertion SQL string
        if args.annotation_type == 'annotation':

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
            if args.username is not None:
                username = usernames[0]
                if args.username not in usernames:
                    print(f'WARNING: username "{args.username}" not found, using "{username}" instead.')
            
            else:
                username = usernames[0]
            
            print(f'Inserting annotations under username "{username}".')

            queryStr = sql.SQL('''
            INSERT INTO {id_anno} (username, image, timeCreated, timeRequired, segmentationmask, width, height)
            VALUES(
                %s,
                (SELECT id FROM {id_img} WHERE filename LIKE %s),
                (TIMESTAMP %s),
                -1,
                %s,
                %s,
                %s
            );''').format(
                id_anno=sql.Identifier(args.project, 'annotation'),
                id_img=sql.Identifier(args.project, 'image')
            )
        elif args.annotation_type == 'prediction':
            queryStr = sql.SQL('''
            INSERT INTO {id_pred} (image, timeCreated, segmentationmask, width, height)
            VALUES(
                (SELECT id FROM {id_img} WHERE filename = %s),
                (TIMESTAMP %s),
                %s,
                %s,
                %s
            );''').format(
                id_pred=sql.Identifier(args.project, 'prediction'),
                id_img=sql.Identifier(args.project, 'image')
            )

    # locate all images and their base names
    print('\nAdding image paths...')
    imgs = {}
    imgFiles = glob.glob(os.path.join(imgBaseDir, '**'), recursive=True)    # os.listdir(imgBaseDir)
    for i in tqdm(imgFiles):
        if os.path.isdir(i):
            continue
        
        basePath, ext = os.path.splitext(i)

        if ext.lower() not in drivers.VALID_IMAGE_EXTENSIONS:
            continue

        baseName = basePath.replace(imgBaseDir, '')
        imgs[baseName] = i.replace(imgBaseDir, '')


    # ignore images that are already in database
    print('Filter images already in database...')
    imgs_filenames = set(imgs.values())
    imgs_existing = dbConn.execute(sql.SQL('''
        SELECT filename FROM {id_img};
    ''').format(id_img=sql.Identifier(args.project, 'image')), None, 'all')
    imgs_existing = set([i['filename'] for i in imgs_existing])

    imgs_filenames = list(imgs_filenames.difference(imgs_existing))
    imgs_filenames = [(i,) for i in imgs_filenames]

    # push image to database
    print('Adding to database...')
    dbConn.insert(sql.SQL('''
        INSERT INTO {id_img} (filename)
        VALUES %s;
    ''').format(id_img=sql.Identifier(args.project, 'image')),
    imgs_filenames)


    # locate all segmentation masks
    if args.label_folder is not None:
        print('\nAdding segmentation masks...')
        labelFiles = glob.glob(os.path.join(args.label_folder, '**'), recursive=True)
        for l in tqdm(labelFiles):

            if os.path.isdir(l) or 'classes.txt' in l:
                continue

            basePath, _ = os.path.splitext(l)
            baseName = basePath.replace(args.label_folder, '')

            # check if matching image exists
            if not baseName in imgs:
                continue

            # load mask
            segMask = Image.open(l)
            sz = segMask.size

            # convert
            dataArray = np.array(segMask).astype(np.uint8)
            b64str = base64.b64encode(dataArray.ravel()).decode('utf-8')

            # add to database
            if args.annotation_type == 'annotation':
                queryArgs = (username, imgs[baseName], currentDT, b64str, sz[0], sz[1])
            else:
                queryArgs = (imgs[baseName], currentDT, b64str, sz[0], sz[1])
            dbConn.execute(queryStr,
                queryArgs)