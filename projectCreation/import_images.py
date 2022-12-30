'''
    Helper function that imports a set of unlabeled images into the database.
    Works recursively (i.e., with images in nested folders) and different file
    formats and extensions (.jpg, .JPEG, .png, etc.).
    Skips images that have already been added to the database.

    Using this script requires the following steps:
    1. Make sure your images are of common format and readable by the web
       server (i.e., convert camera RAW images first).
    2. Call the script from the AIDE code base on the FileServer instance.

    2019-22 Benjamin Kellenberger
'''

import os
import argparse
from psycopg2 import sql

from modules.DataAdministration.backend.dataWorker import DataWorker
from util.helpers import list_directory
from util import drivers


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import images and/or annotations into database, with optional tiling.')
    parser.add_argument('--project', type=str,
                    help='Shortname of the project to insert the images into.')
    parser.add_argument('--user', type=str,
                    help='User account name to register imported data to.')
    parser.add_argument('--image-folder', type=str,
                    help='Base folder on the server of the images to import into the project.')
    parser.add_argument('--split-images', type=int, default=0,
                    help='Set to 1 to enable splitting of images into tiles (default: 0).')
    parser.add_argument('--split-images', type=int, default=0,
                    help='Set to 1 to enable splitting of images into tiles (default: 0).')
    parser.add_argument('--patch-size', type=int, nargs='+',
                    help='Size of split tiles in pixels. Can either be a single int (square patches) or two values for (width, height).')
    parser.add_argument('--stride', type=int, nargs='+', default=-1,
                    help='Stride of the tiles. Can be a single int (same stride in width and height), two values (width, height), or -1 (default: stride equal to patch size).')
    parser.add_argument('--tight', type=int, default=0,
                    help='Set to 1 to limit tiles to image bounds, affecting the stride of the rightmost column and bottommost row of patches (default: 0).')
    parser.add_argument('--upload-images', type=int, default=1,
                    help='Set to 0 to not import images, but just annotations (default: 1 = import images)')
    parser.add_argument('--upload-annotations', type=int, default=1,
                    help='Set to 0 to not import annotations (if available), but just images (default: 1 = import annotations)')
    args = parser.parse_args()

    # setup
    print('Setup...')
    from tqdm import tqdm
    import datetime
    from util.configDef import Config
    from modules import Database
    drivers.init_drivers()

    currentDT = datetime.datetime.now()
    currentDT = '{}-{}-{} {}:{}:{}'.format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)

    config = Config()
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')
    project = args.project


    # check if running on file server
    imgBaseDir = os.path.join(config.getProperty('FileServer', 'staticfiles_dir'), args.project)
    if not os.path.isdir(imgBaseDir):
        raise Exception(f'"{imgBaseDir}" is not a valid directory on this machine. Are you running the script from the file server?')

    
    assert not os.path.samefile(imgBaseDir, args.image_folder), \
        'Error: "--image-folder" cannot be the same as AIDE\'s project folder'


    # locate all files in folder
    print('Locating files...')
    files = list(list_directory(args.image_folder, recursive=True, images_only=False))

    print('Creating import session...')
    dw = DataWorker(config, dbConn, True)
    dw.createUploadSession(project,
                            args.user,
                            len(files),
                            uploadImages=True
                            )

    imgs = set()
    imgFiles = list_directory(args.image_folder, recursive=True)
    imgFiles = list(imgFiles)
    for i in tqdm(imgFiles):
        if os.path.isdir(i):
            continue
        
        _, ext = os.path.splitext(i)
        if ext.lower() not in drivers.VALID_IMAGE_EXTENSIONS:
            continue

        baseName = i.replace(imgBaseDir, '')

        #TODO: check with tiling, file format conversion, etc.

        imgs.add(baseName)

    # ignore images that are already in database
    print('Filter images already in database...')
    imgs_existing = dbConn.execute(sql.SQL('''
        SELECT filename FROM {};
    ''').format(sql.Identifier(project, 'image')), None, 'all')
    if imgs_existing is not None:
        imgs_existing = set([i['filename'] for i in imgs_existing])
    else:
        imgs_existing = set()

    imgs = list(imgs.difference(imgs_existing))
    imgs = [(i,) for i in imgs]

    # push image to database
    print('Adding to database...')
    dbConn.insert(sql.SQL('''
        INSERT INTO {} (filename)
        VALUES %s;
    ''').format(sql.Identifier(project, 'image')),
    imgs)

    print('Done.')