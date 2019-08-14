'''

    2019 Benjamin Kellenberger
'''

import os
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse YOLO annotations and import into database.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Directory of the settings.ini file used for this machine (default: "config/settings.ini").')
    parser.add_argument('--label_folder', type=str, default='/datadrive/landcover/patches_800x600/labels', const=1, nargs='?',
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
    from io import BytesIO
    from util.configDef import Config
    from modules import Database

    if args.label_folder == '':
        args.label_folder = None

    if args.label_folder is not None and not args.label_folder.endswith('/'):
        args.label_folder += '/'

    currentDT = datetime.datetime.now()
    currentDT = '{}-{}-{} {}:{}:{}'.format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)

    config = Config()
    dbConn = Database(config)
    if dbConn.connectionPool is None:
        raise Exception('Error connecting to database.')
    dbSchema = config.getProperty('Database', 'schema')
    
    valid_image_extensions = (
        '.jpg',
        '.jpeg',
        '.png',
        '.gif',
        '.tif',
        '.tiff',
        '.bmp',
        '.ico',
        '.jfif',
        '.pjpeg',
        '.pjp'
    )


    # check if running on file server
    imgBaseDir = config.getProperty('FileServer', 'staticfiles_dir')
    if not os.path.isdir(imgBaseDir):
        raise Exception('"{}" is not a valid directory on this machine. Are you running the script from the file server?'.format(imgBaseDir))

    if args.label_folder is not None and not os.path.isdir(args.label_folder):
        raise Exception('"{}" is not a valid directory on this machine.'.format(args.label_folder))

    if not imgBaseDir.endswith('/'):
        imgBaseDir += '/'


    # parse class names and indices
    if args.label_folder is not None:
        with open(os.path.join(args.label_folder, 'classes.txt'),'r') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            className = line.strip()            

            # push to database
            dbConn.execute('''
                INSERT INTO {}.LABELCLASS (name, idx)
                VALUES (
                    %s, %s
                )
                ON CONFLICT (name) DO NOTHING;
            '''.format(dbSchema),
            (className,idx,))

        # prepare insertion SQL string
        if args.annotation_type == 'annotation':
            sql = '''
            INSERT INTO {}.ANNOTATION (username, image, timeCreated, timeRequired, segmentationmask, width, height)
            VALUES(
                '{}',
                (SELECT id FROM {}.IMAGE WHERE filename LIKE %s),
                (TIMESTAMP %s),
                -1,
                %s,
                %s,
                %s
            );'''.format(dbSchema, config.getProperty('Project', 'adminName'), dbSchema)
        elif args.annotation_type == 'prediction':
            sql = '''
            INSERT INTO {}.PREDICTION (image, timeCreated, segmentationmask, width, height)
            VALUES(
                (SELECT id FROM {}.IMAGE WHERE filename = %s),
                (TIMESTAMP %s),
                %s,
                %s,
                %s
            );'''.format(dbSchema, dbSchema)

    # locate all images and their base names
    print('\nAdding image paths...')
    imgs = {}
    imgFiles = glob.glob(os.path.join(imgBaseDir, '**'), recursive=True)    # os.listdir(imgBaseDir)
    for i in tqdm(imgFiles):
        if os.path.isdir(i):
            continue
        
        basePath, ext = os.path.splitext(i)

        if ext.lower() not in valid_image_extensions:
            continue

        baseName = basePath.replace(imgBaseDir, '')
        imgs[baseName] = i.replace(imgBaseDir, '')


    # ignore images that are already in database
    print('Filter images already in database...')
    imgs_filenames = set(imgs.values())
    imgs_existing = dbConn.execute('''
        SELECT filename FROM {}.image;
    '''.format(dbSchema), None, 'all')
    imgs_existing = set([i['filename'] for i in imgs_existing])

    imgs_filenames = list(imgs_filenames.difference(imgs_existing))
    imgs_filenames = [(i,) for i in imgs_filenames]

    # push image to database
    print('Adding to database...')
    dbConn.insert('''
        INSERT INTO {}.image (filename)
        VALUES %s;
    '''.format(dbSchema),
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
            dbConn.execute(sql,
                (imgs[baseName], currentDT, b64str, sz[0], sz[1]))