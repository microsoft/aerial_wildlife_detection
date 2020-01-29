'''
    Helper function that checks the database for image entries that do not
    exist anymore on the file system, and removes them accordingly.
    
    WARNING: this also removes any annotations, predictions, statistics, and
    viewcounts for the respective entry. No warning prompt will be posed,
    and any images not found on the file system, together with their data,
    will be lost forever.
    Please use this script with care.

    2020 Benjamin Kellenberger
'''

import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check the database for orphaned image entries and remove them.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    args = parser.parse_args()
    

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    import glob
    from tqdm import tqdm
    import datetime
    from PIL import Image
    from util.configDef import Config
    from modules import Database


    config = Config()
    dbConn = Database(config)
    if dbConn.connectionPool is None:
        raise Exception('Error connecting to database.')
    dbSchema = config.getProperty('Database', 'schema')


    # check if running on file server
    imgBaseDir = config.getProperty('FileServer', 'staticfiles_dir')
    if not os.path.isdir(imgBaseDir):
        raise Exception('"{}" is not a valid directory on this machine. Are you running the script from the file server?'.format(imgBaseDir))

    if not imgBaseDir.endswith('/'):
        imgBaseDir += '/'


    # get all image paths from the database
    print('Checking database for image entries...')
    imgs_db= dbConn.execute('''
        SELECT filename FROM {}.image;
    '''.format(dbSchema), None, 'all')
    imgs_existing = set([i['filename'] for i in imgs_db])

    
    # locate all images and their base names on the file system
    print('Locating image paths...')
    imgs_files = set()
    imgFiles = glob.glob(os.path.join(imgBaseDir, '**'), recursive=True)
    for i in tqdm(imgFiles):
        if os.path.isdir(i):
            continue

        baseName = i.replace(imgBaseDir, '')
        imgs_files.add(baseName)


    # filter orphaned images
    imgs_orphaned = imgs_existing.difference(imgs_files)
    imgs_orphaned = tuple(imgs_orphaned)

    if len(imgs_orphaned):
        print('Found {} orphaned image entries in database.'.format(len(imgs_orphaned)))

        print('Removing database entries...')
        dbConn.execute('''
            DELETE FROM {dbSchema}.annotation
            WHERE image IN (
                SELECT id FROM {dbSchema}.image
                WHERE filename IN %s
            );
            DELETE FROM {dbSchema}.prediction
            WHERE image IN (
                SELECT id FROM {dbSchema}.image
                WHERE filename IN %s
            );
            DELETE FROM {dbSchema}.image_user
            WHERE image IN (
                SELECT id FROM {dbSchema}.image
                WHERE filename IN %s
            );
            DELETE FROM {dbSchema}.image
            WHERE filename IN %s;
        '''.format(dbSchema=dbSchema),
        (imgs_orphaned, imgs_orphaned, imgs_orphaned, imgs_orphaned), None
        )

    else:
        print('No orphaned images found.')

    print('Done.')
