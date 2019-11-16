'''
    Helper function that imports YOLO labels into the database.
    Needs to be run on the file server.
    This assumes that the images of the dataset have already been placed in the
    folder of the connected file server (parameter "staticfiles_dir" in section [FileServer]
    of .ini file), and that the labels provided here match with a simple file directory lookup.
    Also pushes images to the database if they are not already in there.

    Inputs:
    - labelFolder: path string for a directory with label files (see conventions below)

    Conventions:
    - Labels must be organized as text files in the "labelFolder", with one text file associated
      to exactly one image file.
    - Image-to-label file association: <file server staticfiles_dir>/<img_name>.jpg corresponds to
      <labelFolder>/<img_name>.txt
    - Label text files contain bounding boxes in YOLO format:
                
            <class index> <x> <y> <width> <height>\n
      with:
        - class index: the number of the class, specified in a file "classes.txt" in the same directory.
          One class name per line; class index is provided implicitly as the line number, starting at zero.
        - x, y: center coordinates of the bounding box, normalized to image width, resp. height
        - width, height: width and height of the bounding box, normalized to image width, resp. height
    - Images without a ground truth simply do not need an associated label text file.
    - Optionally, label text files may contain confidences in class order appended to each box. In this case,
      bounding boxes are treated as predictions (instead of annotations) and inserted into the predictions table
      accordingly.
    - Flag "annotationType" specifies the target table into which the provided bounding boxes are inserted.
      May either be "annotation" or "prediction".
    - The optional flag "al_criterion" will specify how to calculate the "priority" value for each prediction,
      if "annotationType" is set to "prediction" and confidence values are provided (see above). It may take one
      of the following values:
          - None: ignore confidences and do not set priority value
          - BreakingTies: calculates the Breaking Ties (Luo, Tong, et al. "Active learning to recognize multiple
            types of plankton." Journal of Machine Learning Research 6.Apr (2005): 589-613.) values for priority
          - MaxConfidence: simply uses the maximum of the confidence scores as a priority value
          - TryAll: tries all heuristics (apart from 'None') and chooses the maximum score over all

    The script then proceeds by parsing the text files and scaling the coordinates back to absolute values,
    which is what will be stored in the database.
    Labels are stored in the "ANNOTATIONS" table, with the following non-standard field
    values:
        - timeCreated: (timestamp of the launch of this script)
        - timeRequired: -1
    Also adds class definitions.

    2019 Benjamin Kellenberger
'''

import os,sys
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse YOLO annotations and import into database.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--label_folder', type=str,
                    help='Directory (absolute path) on this machine that contains the YOLO label text files.')
    parser.add_argument('--annotation_type', type=str, default='annotation', const=1, nargs='?',
                    help='Kind of the provided annotations. One of {"annotation", "prediction"} (default: annotation)')
    parser.add_argument('--al_criterion', type=str, default='TryAll', const=1, nargs='?',
                    help='Criterion for the priority field. One of {"BreakingTies", "MaxConfidence", "TryAll"} (default: TryAll)')
    args = parser.parse_args()
    

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)
    
    import glob
    from tqdm import tqdm
    import yaml
    from PIL import Image
    from util.configDef import Config
    from modules import Database
    from datetime import datetime
    import pytz

    if args.label_folder == '':
        args.label_folder = None

    if args.label_folder is not None and not args.label_folder.endswith('/'):
        args.label_folder += '/'

    currentDT = datetime.now()
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

    print(imgBaseDir)



    # parse class names and indices
    if args.label_folder is not None:
        classdef = {}
        classList = []

        with open(os.path.join(args.label_folder, 'classes.txt'),'r') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            className = line.strip()            
            classList.append(className)

            # push to database
            dbConn.execute('''
                INSERT INTO {}.LABELCLASS (name)
                VALUES (
                    %s
                )
                ON CONFLICT (name) DO NOTHING;
            '''.format(dbSchema),
            (className,))

            # get newly assigned index
            returnVal = dbConn.execute('''
                SELECT id FROM {}.LABELCLASS WHERE name LIKE %s'''.format(dbSchema),
            (className+'%',),
            1)
            classdef[idx] = returnVal[0]['id']


        # prepare insertion SQL string
        if args.annotation_type == 'annotation':
            sql = '''
            INSERT INTO {}.ANNOTATION (username, image, timeCreated, timeRequired, label, x, y, width, height)
            VALUES(
                '{}',
                (SELECT id FROM {}.IMAGE WHERE filename = %s),
                (TIMESTAMP %s),
                -1,
                %s,
                %s,
                %s,
                %s,
                %s
            );'''.format(dbSchema, config.getProperty('Project', 'adminName'), dbSchema)
        elif args.annotation_type == 'prediction':
            sql = '''
            INSERT INTO {}.PREDICTION (image, timeCreated, label, confidence, x, y, width, height, priority)
            VALUES(
                (SELECT id FROM {}.IMAGE WHERE filename = %s),
                (TIMESTAMP %s),
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );'''.format(dbSchema, dbSchema)

    # locate all images and their base names
    print('\nAdding image paths...')
    imgs = {}
    imgFiles = glob.glob(os.path.join(imgBaseDir, '**'), recursive=True)    # os.listdir(imgBaseDir)
    #imgFiles = glob.glob(os.path.join(imgBaseDir, '**'), recursive=False)    # os.listdir(imgBaseDir)
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

    
    # locate all label files
    if args.label_folder is not None:
#    if False:
        print('\nLoading labels file...')
        with open(os.path.join(args.label_folder, 'annotations-species.yml'),'r') as f:
            all_imgs = yaml.load(f)
        print('\nAdding labels...')

        for im in tqdm(all_imgs):
            filename = im['filename']
            filename = filename.replace(" ", "_")
            filename = 'yolo/' + filename

            insert_sql = '''
            INSERT INTO {}.image_user (username, image, viewcount, last_checked, last_time_required)
            VALUES(
                    '{}',
                    (SELECT id FROM {}.IMAGE WHERE filename = %s),
                    %s,
                    %s,
                    %s
            )
            ON CONFLICT (username, image) DO UPDATE SET viewcount = image_user.viewcount + 1, last_checked = EXCLUDED.last_checked, last_time_required = EXCLUDED.last_time_required
            ;'''.format(dbSchema, config.getProperty('Project', 'adminName'), dbSchema)

            lastChecked = datetime.now(tz=pytz.utc)
            lastTimeRequired = 0

            dbConn.execute(insert_sql, (filename, 1, lastChecked, lastTimeRequired))
            if im['object']:
                w = im['width']
                h = im['height']

                for obj in im['object']:

                    label = obj['name']
                    xc = 0.5*(obj['xmax']+obj['xmin'])/w
                    xd = (obj['xmax']-obj['xmin'])/w
                    yc = 0.5*(obj['ymax']+obj['ymin'])/h
                    yd = (obj['ymax']-obj['ymin'])/h
                    dbConn.execute(sql,
                        (filename, currentDT, classdef[classList.index(label)], xc, yc, xd, yd))
                    # viewcount table



