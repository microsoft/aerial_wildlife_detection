'''
    Helper function that imports a set of unlabeled images into the database.
    Works recursively (i.e., with images in nested folders) and different file
    formats and extensions (.jpg, .JPEG, .png, etc.).
    Skips images that have already been added to the database.

    Using this script requires the following steps:
    1. Make sure your images are of common format and readable by the web
       server (i.e., convert camera RAW images first).
    2. Copy your image folder into the FileServer's root file directory (i.e.,
       corresponding to the path under "staticfiles_dir" in the configuration
       *.ini file).
    3. Call the script from the AIDE code base on the FileServer instance.

    2019-22 Benjamin Kellenberger
'''

import os
import sys
import argparse
import shutil
from datetime import datetime
from psycopg2 import sql
from PIL import Image

from util.helpers import VALID_IMAGE_EXTENSIONS, listDirectory
from util.imageSharding import split_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Import images into database.')
    parser.add_argument('--project', type=str, default="kuzikus",
                    help='Shortname of the project to insert the images into.')
    parser.add_argument('--settings_filepath', type=str, default='config/settings.ini', const=1, nargs='?',
                    help='Manual specification of the directory of the settings.ini file; only considered if environment variable unset (default: "config/settings.ini").')
    parser.add_argument('--folder', type=str, nargs='?',
                    help='Optional directory to import images from. If not specified, the project folder within AIDE is parsed. Must be specified and a distinct folder when image splitting into patches is enabled.')
    parser.add_argument('--split', type=int, default=0,
                    help='Set to 1 to enable splitting of images into patches (default: 0 = do not split)')
    parser.add_argument('--patch-size', type=int, nargs='?',
                    help='Image patch size in pixels. Image is split into square patches when one value is provided, or else a tuple of (width, height) can be given. Ignored when "--split" is 0.')
    parser.add_argument('--stride', type=int, nargs='?',
                    help='Image patch stride (offset w.r.t. previous, i.e., left and/or top patch). Can be one value (equal stride in horizontal and vertical direction), two values (horizontal, vertical), or omitted (stride equal to "--patch-size"). Ignored when "--split" is 0.')
    parser.add_argument('--tight', type=int, default=1,
                    help='Set to 1 to not exceed patches beyond image bounds (default: 1 = true)')
    args = parser.parse_args()

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    from tqdm import tqdm
    from util.configDef import Config
    from modules import Database

    config = Config()
    dbConn = Database(config)
    if not dbConn.canConnect():
        raise Exception('Error connecting to database.')
    project = args.project

    # arguments
    projectFolder = os.path.join(config.getProperty('FileServer', 'staticfiles_dir'), project)
    assert os.path.exists(projectFolder), f'"{projectFolder}" is not a valid directory on this machine. Are you running the script from the file server?'
    imgFolder = args.folder
    if imgFolder is None:
        imgFolder = projectFolder
    assert os.path.exists(imgFolder), f'Image folder "{imgFolder}" does not exist'

    if not imgFolder.endswith(os.sep):
        imgFolder += os.sep
    do_split = bool(args.split)
    if do_split:
        assert args.folder != projectFolder, f'"--folder" must be set and must not be equal to the project\'s image directory when "--split" is enabled'
        patchSize = args.patch_size
        if isinstance(patchSize, int):
            patchSize = (patchSize, patchSize)
        patchSize = list(patchSize)[:2]
        assert patchSize[0] > 0 and patchSize[1] > 0, f'Invalid patch size specified ({patchSize} <= 0)'
        stride = args.stride
        if stride is None:
            stride = patchSize
        elif isinstance(stride, int):
            stride = (stride, stride)
        stride = list(stride)[:2]
        assert stride[0] > 0 and stride[1] > 0, f'Invalid stride specified ({stride} <= 0)'
        tight = bool(args.tight)
    
    # locate all images and their base names
    print('Locating images...')
    imgs = {}
    imgFiles = listDirectory(imgFolder, recursive=True)    #glob.glob(os.path.join(imgBaseDir, '**'), recursive=True)  #TODO: check if correct
    imgFiles = list(imgFiles)
    for i in tqdm(imgFiles):
        if os.path.isdir(i):
            continue
        
        _, ext = os.path.splitext(i)
        if ext.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        baseName = i.replace(imgFolder, '')
        if do_split:
            # look for patterns of type '<img name>%<ext>', e.g. 'image_800_600.jpg'
            queryName, ext = os.path.splitext(baseName)
            queryName = queryName + '%' + ext
            selectStr = 'DISTINCT REGEXP_REPLACE(filename, \'\_[0-9]+\_[0-9]+\.\', \'.\') AS filename'
        else:
            # no splitting; image name is the same in the database
            queryName = baseName
            selectStr = 'filename'
        imgs[baseName] = queryName

    # ignore images that are already in database
    print('Filter images already in database...')
    imgs_existing = dbConn.execute(sql.SQL('''
        SELECT {}
        FROM {}
        WHERE filename LIKE ANY(%s)
    ''').format(
        sql.SQL(selectStr),
        sql.Identifier(project, 'image')),
    (list(imgs.values()),), 'all'
    )
    if imgs_existing is not None and len(imgs_existing):
        imgs_existing = set([i['filename'] for i in imgs_existing])
    else:
        imgs_existing = set()

    imgs = list(imgs.difference(imgs_existing))

    if not len(imgs):
        print('WARNING: no images found, or else all images are already registered in database. Aborting...')
        sys.exit(0)

    # log file
    currentDT = datetime.now()
    logFilePath = 'aide_image_import_{}-{}-{}_{}_{}_{}.log'.format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)
    logFile = open(logFilePath, 'w')
    logFile.write('file_name;;errors\n')

    try:
        if imgFolder == projectFolder:
            # push image to database
            imgs = [(i,) for i in imgs]
            print('Registering images in database...')
            dbConn.insert(sql.SQL('''
                INSERT INTO {} (filename)
                VALUES %s;
            ''').format(sql.Identifier(project, 'image')),
            imgs)
            for img in imgs:
                logFile.write(img + ';;0\n')
        
        elif do_split:
            print('Splitting images into tiles and saving to project folder...')
            imgs_final = []
            for img in tqdm(imgs):
                try:
                    with Image.open(os.path.join(imgFolder, img)) as image:
                        images, coords = split_image(image,
                                                patchSize,
                                                stride,
                                                tight)
                        bareFileName, ext = os.path.splitext(img)
                        for i in range(len(images)):
                            try:
                                patchFilename = f'{bareFileName}_{coords[i][0]}_{coords[i][1]}{ext}'
                                patchFilepath = os.path.join(projectFolder, patchFilename)
                                if os.path.exists(patchFilepath):
                                    raise Exception('file already exists in project folder')
                                parent, _ = os.path.split(patchFilepath)
                                os.makedirs(parent, exist_ok=True)
                                images[i].save(patchFilepath)
                                imgs_final.append(patchFilename)
                            except Exception as e:
                                logFile.write(f'{patchFilename};;{str(e)}\n')

                except Exception as e:
                    logFile.write(f'{img};;{str(e)}\n')

            if len(imgs_final):
                print(f'Split {len(imgs)} into {len(imgs_final)} patches; importing to database...')
                dbConn.insert(sql.SQL('''
                    INSERT INTO {} (filename)
                    VALUES %s;
                ''').format(sql.Identifier(project, 'image')),
                [(i,) for i in imgs_final])
                for img in imgs_final:
                    logFile.write(f'{img};;0\n')
            else:
                print('WARNING: no image patches created.')

        else:
            print('Importing images into project folder...')
            imgs_final = []
            for img in tqdm(imgs):
                try:
                    destFilePath = os.path.join(projectFolder, img)
                    if os.path.exists(destFilePath):
                        raise Exception('file already exists in project folder')
                    parent, _ = os.path.split(destFilePath)
                    os.makedirs(parent, exist_ok=True)
                    shutil.copyfile(
                        os.path.join(imgFolder, img),
                        destFilePath
                    )
                    imgs_final.append(img)
                except Exception as e:
                    logFile.write(f'{img};;{str(e)}\n')
            
            if len(imgs_final):
                print(f'{len(imgs_final)} images copied into project folder; importing to database...')
                dbConn.insert(sql.SQL('''
                    INSERT INTO {} (filename)
                    VALUES %s;
                ''').format(sql.Identifier(project, 'image')),
                [(i,) for i in imgs_final])
                for img in imgs_final:
                    logFile.write(f'{img};;0\n')
            else:
                print('WARNING: no image patches created.')
    finally:
        logFile.close()

    print(f'Done. Log file written to {logFilePath}.')