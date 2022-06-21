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
        self.maxIdx = 0 if not len(lcIDs) else max([l['idx'] for l in lcIDs])           # current max label class ordinal idx; cannot go beyond 255


    @classmethod
    def get_html_options(cls, method):
        if method == 'import':
            return '''
            <input type="checkbox" id="verify_image_size" />
            <label for="verify_image_size">verify image size</label>
            <p style="margin-left:10px;font-style:italic">
                If checked, segmentation maps will be checked for size
                and compared against target image sizes.
                This may be more accurate, but significantly slower.
            </p>
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
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, empty segmentation masks will be skipped

        now = helpers.current_time()

        importedAnnotations, warnings, errors = defaultdict(list), [], []

        # get potentially valid segmentation files
        segFiles = self._get_segmentation_images(fileDict, self.tempDir)
        if not len(segFiles):
            return {
                'result': {},
                'warnings': [],
                'errors': ['No valid segmentation images found.']
            }
        
        # project labelclass look-up tables
        labelclasses_new = set()
        labelclasses_dropped = set()        # label classes that could not be considered, either due to max idx ordinal being reached or skipping

        lcSet_idx = set(l['idx'] for l in self.labelClasses.values())
        lcLUT_color = {}
        labelclasses_update = set()
        for lcID in self.labelClasses.keys():
            color = self.labelClasses[lcID].get('color', None)
            if color is None:
                # we don't allow empty colors for segmentation projects anymore; flag for update
                labelclasses_update.add(lcID)
            else:
                color = helpers.hexToRGB(color)
                lcLUT_color[color] = self.labelClasses[lcID]['idx']

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
            if file not in fKeys or fKeys[file] not in img_lut:
                warnings.append(
                    f'Annotation file "{file}": no equivalent image found in AIDE project.'
                )
                continue

            # load map
            arr = drivers.load_from_disk(os.path.join(self.tempDir, file))

            # get unique classes across channel dimension
            numBands = arr.shape[0]
            if numBands == 1:
                # segmask is already index-based
                arr_out = np.copy(arr).squeeze().astype(np.uint8)

                # check for non-existing classes
                classIdx_new = lcSet_idx.difference(set(np.unique(arr)))
                labelclasses_new.add(classIdx_new)

                if skipUnknownClasses:
                    # set new classes to zero
                    for cl in classIdx_new:
                        arr_out[arr_out==cl] = 0
                else:
                    # check if new classes still allowed as per max idx ordinal serial
                    for cl in classIdx_new:
                        if cl not in labelclasses_new:
                            if self.maxIdx >= 255:
                                # maximum reached
                                labelclasses_dropped.add(cl)
                                arr_out[arr_out==cl] = 0
                            else:
                                # still capacity; add
                                self.maxIdx = max(self.maxIdx, cl)
            
            else:
                # RGB; convert from colors
                arr_flat = np.reshape(arr, (arr.shape[0], -1))
                arr_colors = np.unique(arr_flat, axis=1)
                arr_colors = set(tuple(a) for a in arr_colors.T.tolist())

                arr_out = np.zeros(shape=arr.shape[1:3], dtype=np.uint8)

                # check for non-existing classes
                colors_new = set(lcLUT_color.keys()).difference(arr_colors)

                for rgb in arr_colors:
                    if skipUnknownClasses and rgb in colors_new:
                        # skip in-painting
                        continue
                    else:
                        if rgb not in lcLUT_color:
                            # new color detected; check if capacity available as per max idx ordinal serial
                            if self.maxIdx >= 255:
                                # maximum reached
                                labelclasses_dropped.add(rgb)
                                continue
                            else:
                                # still capacity; add
                                idx = self.maxIdx + 1
                                labelclasses_new.add(idx)
                                lcLUT_color[rgb] = idx
                                self.maxIdx += 1

                        idx = lcLUT_color[rgb]
                        valid = (arr[0,...] == rgb[0]) * (arr[1,...] == rgb[1]) * (arr[2,...] == rgb[2])
                        arr_out[valid] = idx

            if not np.sum(arr_out):
                warnings.append(
                    f'Annotation file "{file}": empty annotations or no valid label classes found.'
                )
                if skipEmptyImages:
                    continue
            
            # register in database, accounting for virtual views
            imgs_insert = []
            imgEntries = img_lut[fKeys[file]]
            for entry in imgEntries:
                if entry.get('x', None) is not None and entry.get('y', None) is not None:
                    # virtual view; crop segmentation mask at given position
                    arr_crop = np.zeros(shape=(entry['width'], entry['height']), dtype=np.uint8)
                    bounds = (
                        min(arr_out.shape[0]-1, max(0, entry['x'])),        # left
                        min(arr_out.shape[1]-1, max(0, entry['y'])),        # top
                        min(arr_out.shape[0], entry['x']+entry['width']),   # right
                        min(arr_out.shape[1], entry['y']+entry['height']),  # bottom
                    )
                    arr_crop[:bounds[2]-bounds[0],:bounds[3]-bounds[1]] = arr_out[bounds[0]:bounds[2], bounds[1]:bounds[3]]
                    b64str = base64.b64encode(arr_crop.ravel()).decode('utf-8')
                    imgs_insert.append((
                        targetAccount,
                        entry['id'],
                        now,
                        -1,
                        False,
                        b64str,
                        arr_crop.shape[0],      #TODO: order
                        arr_crop.shape[1]
                    ))
                else:
                    # no virtual view
                    b64str = base64.b64encode(arr_out.ravel()).decode('utf-8')
                    imgs_insert.append((
                        targetAccount,
                        entry['id'],
                        now,
                        -1,
                        False,
                        b64str,
                        arr_out.shape[0],       #TODO: order
                        arr_out.shape[1]
                    ))
                
            # insert
            if len(imgs_insert):
                dbInsert = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, timeCreated, timeRequired, unsure, segmentationMask, width, height)
                    VALUES %s
                    RETURNING image, id;
                ''').format(sql.Identifier(self.project, 'annotation')),
                tuple(imgs_insert), 'all')
                for row in dbInsert:
                    importedAnnotations[row[0]].append(row[1])

        # register new / update existing label classes
        if (not skipUnknownClasses and len(labelclasses_new)) or len(labelclasses_update):
            lcColors = set(lcLUT_color.keys())
            lcVals = []
            if not skipUnknownClasses:
                for l in labelclasses_new:
                    color = helpers.randomHexColor(lcColors)
                    lcColors.add(color)
                    lcVals.append((f'Class {l}', l, color))
                
            # add color for existing classes to be updated
            for lcID in labelclasses_update:
                color = helpers.randomHexColor(lcColors)
                lcColors.add(color)
                lcName = self.labelClasses[lcID]['name']
                lcIdx = self.labelClasses[lcID]['idx']
                lcVals.append((lcName, lcIdx, color))
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (name, idx, color)
                VALUES %s
                ON CONFLICT (idx) DO
                UPDATE SET color = EXCLUDED.color;
            ''').format(sql.Identifier(self.project, 'labelclass')),
            tuple(lcVals))

        if len(labelclasses_dropped):
            warnings.append('The following label class indices and/or RGB colors were identified but could not be added to the project: {}.\nPixels annotated with those classes were set to zero.'.format(
                ', '.join(str(l) for l in labelclasses_dropped)
            ))

        # also set in image_user relation
        if len(importedAnnotations):
            imgIDs_added = list(importedAnnotations.keys())
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (username, image, last_checked, last_time_required)
                VALUES %s
                ON CONFLICT (username, image) DO UPDATE
                SET last_time_required = -1;
            ''').format(sql.Identifier(self.project, 'image_user')),
            tuple([(targetAccount, i, now, -1) for i in imgIDs_added]))

            if markAsGoldenQuestions:
                self.dbConnector.insert(sql.SQL('''
                    UPDATE {}
                    SET isGoldenQuestion = TRUE
                    WHERE id IN (%s);
                ''').format(sql.Identifier(self.project, 'image')),
                tuple([(i,) for i in imgIDs_added]))

        return {
            'result': dict(importedAnnotations),
            'warnings': warnings,
            'errors': errors
        }
    

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
