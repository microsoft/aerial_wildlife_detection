'''
    Parser for pixel-wise segmentation rasters.

    2022 Benjamin Kellenberger
'''

import os
import re
import shutil
import base64
from collections import defaultdict
import numpy as np
from psycopg2 import sql
import rasterio
from rasterio.windows import Window
from util.drivers.imageDrivers import GDALImageDriver

from util.parsers.abstractParser import AbstractAnnotationParser
from util import helpers, drivers
drivers.init_drivers()


class SegmentationFileParser(AbstractAnnotationParser):

    NAME = 'Image Files'
    INFO = '<p>Supports pixel-wise annotations in images (e.g., TIFFs)'
    ANNOTATION_TYPES = ('segmentationMasks')

    FILE_SUB_PATTERN = '(\/|\\\\)*.*\/*(images|labels|annotations)(\/|\\\\)*'           # pattern that attempts to identify image file name from label file name


    def _init_labelclasses(self):
        '''
            Override of parent: we also need the class indices and colors here.
        '''
        # create project label class LUT
        lcIDs = self.dbConnector.execute(sql.SQL('''
            SELECT id, name, idx, color
            FROM {};
        ''').format(sql.Identifier(self.project, 'labelclass')),
        None, 'all')
        self.labelClasses = dict([[l['id'], {'name':l['name'], 'idx': l['idx'], 'color': l['color']}] for l in lcIDs])


    @classmethod
    def get_html_options(cls, method):
        if method == 'import':
            return '''
            
            '''
        
        else:
            return '''
            <div>
                <fieldset>
                    <legend>Label class encoding:</legend>
                    <input type="radio" id="export_indices" name="export_colors" checked="checked" />
                    <label for="export_indices">single-band images with label class indices</label>
                    <br />
                    <input type="radio" id="export_colors" name="export_colors" />
                    <label for="export_colors">RGB images with label class colors</label>
                </fieldset>
            </div>
            '''
    

    @classmethod
    def _get_segmentation_images(cls, fileDict, folderPrefix):
        '''
            Iterates through keys of a provided dict of file names and returns
            all those that are one- or three-band images.
        '''
        segFiles = []
        for file_orig in fileDict.keys():
            if isinstance(folderPrefix, str):
                filePath = os.path.join(folderPrefix, file_orig)
            else:
                filePath = file_orig
            if not os.path.isfile(filePath) and not os.path.islink(filePath):
                continue
            try:
                sz = drivers.GDALImageDriver.size(filePath)
                if sz[0] in (1,3):
                    segFiles.append(file_orig)
            except:
                # unparseable
                continue
        return segFiles


    @classmethod
    def is_parseable(cls, fileDict, folderPrefix):
        '''
            File dict is parseable if at least one file is a one- or three-band
            image.
        '''
        return len(cls._get_segmentation_images(fileDict, folderPrefix)) > 0
    

    def import_annotations(self, fileDict, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):

        # args setup
        verifyImageSize = kwargs.get('verify_image_size', False)    # if True, image sizes will be retrieved from files, not just from COCO metadata

        now = helpers.current_time()

        # get potentially valid segmentation files
        segFiles = self._get_segmentation_images(fileDict, self.tempDir)
        if not len(segFiles):
            return {
                'result': {},
                'warnings': [],
                'errors': ['No valid segmentation images found.']
            }
        
        labelclasses_new = []

        # find segmentation files that have a corresponding image registered
        fKeys = {}
        for file in segFiles:
            # file key to match the annotation file with the image(s) present in the database
            fKey = file
            if fKey.startswith(os.sep) or fKey.startswith('/'):
                fKey = fKey[1:]
            fKey = re.sub(self.FILE_SUB_PATTERN, '', fKey, flags=re.IGNORECASE)        # replace "<base folder>/(labels|annotations)/*" with "%/*" for search in database with "LIKE" expression
            fKey = os.path.splitext(fKey)[0]
            fKeys[file] = fKey

        # find corresponding images in database, blacklisting those that had annotations imported before
        dbQuery = self.dbConnector.execute(sql.SQL('''
            WITH labelquery(labelname) AS (
                VALUES {pl}
            )
            SELECT id, filename, COALESCE(x,0) AS x, COALESCE(y,0) AS y, width, height, labelname
            FROM (
                SELECT id, filename, x, y, width, height
                FROM {id_img}
                WHERE id NOT IN (
                    SELECT image
                    FROM {id_iu}
                    WHERE last_time_required < 0
                )
            ) AS q
            JOIN labelquery
            ON filename ILIKE CONCAT('%%', labelname, '%%');
        ''').format(
            pl=sql.SQL(','.join(['%s' for _ in fKeys.keys()])),
            id_img=sql.Identifier(self.project, 'image'),
            id_iu=sql.Identifier(self.project, 'image_user')
        ),
        tuple((v,) for v in fKeys.values()), 'all')
        img_lut = defaultdict(list)
        for row in dbQuery:
            if verifyImageSize or row['width'] is None or row['height'] is None:
                fname = os.path.join(self.projectRoot, row['filename'])
                driver = drivers.get_driver(fname)
                imsize = driver.size(fname)
                row['width'] = imsize[2]
                row['height'] = imsize[1]
            img_lut[row['labelname']].append(row)

        # filter valid images: found in database, with valid annotation, and not blacklisted
        for file in segFiles:
            # load map
            arr = drivers.load_from_disk(file)

            # get unique classes across channel dimension
            arrClasses = np.unique(np.reshape(arr, (arr.shape[0], -1)), 0)

            print('debug')
    

    def export_annotations(self, annotations, destination, **kwargs):

        # args setup
        exportColors = kwargs.get('export_colors', False)      # if True, RGB images with labelclass colors will be exported instead of single channel index maps

        now = helpers.slugify(helpers.current_time())

        # export labelclass definition
        lcStr = 'labelclass,index,color\n'
        for lc in self.labelClasses.values():
            lcStr += '{},{},{}\n'.format(
                lc['name'], lc['idx'], lc['color']
            )
        destination.writestr('classes.txt', data=lcStr)

        # create labelclass color lookup table if needed
        if exportColors:
            colorLUT = {}
            for lcDef in self.labelClasses.values():
                idx = lcDef['idx']
                try:
                    color = helpers.hexToRGB(lcDef['color'])
                except:
                    # error parsing color; this should never happen
                    color = 0
                colorLUT[idx] = color

        # create annotation lookup table
        anno_lut = defaultdict(list)
        for row in annotations['annotations']:
            anno_lut[row['image']].append(row)

        # create image lookup table for virtual views
        img_lut = defaultdict(list)
        for row in annotations['images']:
            img_lut[row['filename']].append({
                'id': row['id'],
                'x': row.get('x') or 0,
                'y': row.get('y') or 0,
                'width': row.get('width'),
                'height': row.get('height')
            })
        
        # iterate over packets of images and write out segmentation data,
        # combining virtual views
        files_exported = []
        for filename in img_lut.keys():
            # get total image file properties
            with rasterio.open(os.path.join(self.projectRoot, filename)) as src:
                sz = [src.count, src.height, src.width]
                transform = src.transform
                crs = src.crs
            
            destFiles = {}      # username: file name

            for imgView in img_lut[filename]:
                imgID = imgView['id']
                fname, _ = os.path.splitext(filename)

                for anno in anno_lut[imgID]:
                    # put potentially multiple segmentation masks into different subfolders
                    username = str(anno.get('username', 'cnnstate'))
                    if username not in destFiles:
                        destFilename = os.path.join('segmentation', username, fname+'.tif')
                        destFiles[username] = destFilename
                        tempDest = os.path.join(self.tempDir, 'segmentation_export', now, destFilename)
                        parent, _ = os.path.split(tempDest)
                        os.makedirs(parent, exist_ok=True)
                    else:
                        destFilename = destFiles[username]

                    # load segmap
                    segmap = self.dbConnector.execute(sql.SQL('''
                        SELECT segmentationMask
                        FROM {}
                        WHERE id = %s;
                    ''').format(sql.Identifier(self.project, annotations['data_type'])),
                    (anno['id'],), 1)

                    # convert base64 mask to image
                    raster_raw = np.frombuffer(base64.b64decode(segmap[0]['segmentationmask']), dtype=rasterio.ubyte)
                    raster_raw = np.reshape(raster_raw, (1, int(anno['height']),int(anno['width']),))

                    # convert to RGB image if needed
                    if exportColors:
                        raster_raw = raster_raw.squeeze()
                        raster = np.zeros((3, *raster_raw.shape[:2]), dtype=rasterio.ubyte)
                        lcIdx = np.unique(raster_raw).tolist()
                        for lc in lcIdx:
                            if lc not in colorLUT:
                                continue
                            val = colorLUT[lc]
                            valid = (raster_raw == lc)
                            for v in range(3):
                                raster[v,valid] = val[v]
                    else:
                        raster = raster_raw

                    # window where to write segmask part into
                    window = Window(
                        imgView['y'],
                        imgView['x'],
                        imgView['height'],
                        imgView['width']
                    )

                    with rasterio.open(
                        tempDest, 'w',
                        driver='GTiff',
                        nodata=0,
                        width=sz[2], height=sz[1], count=3 if exportColors else 1,
                        transform=transform, crs=crs,
                        dtype=rasterio.ubyte) as dst:
                        dst.write(raster, window=window, indexes=list(range(1,len(raster)+1)))
                    
            # move all temp files into zipfile
            for key in destFiles:
                destFilename = destFiles[key]
                tempDest = os.path.join(self.tempDir, 'segmentation_export', now, destFilename)
                destination.write(tempDest, destFilename)
                os.remove(tempDest)
                files_exported.append(destFilename)

        shutil.rmtree(os.path.join(self.tempDir, 'segmentation_export', now))

        return {
            'files': tuple(files_exported)
        }
