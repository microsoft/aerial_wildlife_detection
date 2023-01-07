'''
    Parser for pixel-wise segmentation rasters.

    2022-23 Benjamin Kellenberger
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
    '''
        Parser for pixel-wise segmentation rasters.
    '''

    NAME = 'Image Files'
    INFO = '<p>Supports pixel-wise annotations in images (e.g., TIFFs)'
    ANNOTATION_TYPES = ('segmentationMasks')

    # pattern that attempts to identify image file name from label file name
    FILE_SUB_PATTERN = r'(\/|\\\\)*.*\/*(images|labels|annotations)(\/|\\\\)*'


    def _init_labelclasses(self):
        '''
            Override of parent: we also need the class indices and colors here.
        '''
        # create project label class LUT
        lc_ids = self.dbConnector.execute(sql.SQL('''
            SELECT id, name, idx, color
            FROM {};
        ''').format(sql.Identifier(self.project, 'labelclass')),
        None, 'all')
        self.labelClasses = dict(
            [l['id'], {'name':l['name'], 'idx': l['idx'], 'color': l['color']}] for l in lc_ids)
        # current max label class ordinal idx; cannot go beyond 255
        self.max_idx = 0 if len(lc_ids) == 0 else max(l['idx'] for l in lc_ids)


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
    def _get_segmentation_images(cls, file_dict: dict, folder_prefix: str) -> list:
        '''
            Iterates through keys of a provided dict of file names and returns all those that are
            one- or three-band images.
        '''
        seg_files = []
        for file_orig in file_dict.keys():
            if isinstance(folder_prefix, str):
                file_path = os.path.join(folder_prefix, file_orig)
            else:
                file_path = file_orig
            if not os.path.isfile(file_path) and not os.path.islink(file_path):
                continue
            try:
                size = drivers.GDALImageDriver.size(file_path)
                if size[0] in (1,3):
                    seg_files.append(file_orig)
            except Exception:
                # unparseable
                continue
        return seg_files


    @classmethod
    def is_parseable(cls, fileDict: dict, folderPrefix: str) -> bool:
        '''
            File dict is parseable if at least one file is a one- or three-band image.
        '''
        return len(cls._get_segmentation_images(fileDict, folderPrefix)) > 0


    def import_annotations(self, fileDict: dict,
                                targetAccount: str,
                                skipUnknownClasses: bool,
                                markAsGoldenQuestions: bool,
                                **kwargs):

        # args setup

        # if True, image sizes will be retrieved from files, not just from COCO metadata
        verify_image_size = kwargs.get('verify_image_size', False)
        # if True, empty segmentation masks will be skipped
        skip_empty_images = kwargs.get('skip_empty_images', False)

        now = helpers.current_time()

        imported_annotations, warnings, errors = defaultdict(list), [], []

        # get potentially valid segmentation files
        seg_files = self._get_segmentation_images(fileDict, self.tempDir)
        if len(seg_files) == 0:
            return {
                'result': {},
                'warnings': [],
                'errors': ['No valid segmentation images found.']
            }

        # project labelclass look-up tables
        labelclasses_new = set()

        # label classes not considered, either due to max idx ordinal being reached or skipping
        labelclasses_dropped = set()

        lc_set_idx = set(l['idx'] for l in self.labelClasses.values())
        lc_lut_color = {}
        labelclasses_update = set()
        for lc_id, lc_meta in self.labelClasses.items():
            color = lc_meta.get('color', None)
            if color is None:
                # we don't allow empty colors for segmentation projects anymore; flag for update
                labelclasses_update.add(lc_id)
            else:
                color = helpers.hexToRGB(color)
                lc_lut_color[color] = lc_meta['idx']

        # find segmentation files that have a corresponding image registered
        file_keys = {}
        for file in seg_files:
            # file key to match the annotation file with the image(s) present in the database
            file_key = file
            if file_key.startswith(os.sep) or file_key.startswith('/'):
                file_key = file_key[1:]
            # replace "<base folder>/(labels|annotations)/*" with "%/*" for search in database with
            # "LIKE" expression
            file_key = re.sub(self.FILE_SUB_PATTERN, '', file_key, flags=re.IGNORECASE)
            file_key = os.path.splitext(file_key)[0]
            file_keys[file] = file_key

        # find corresponding images in database, blacklisting those with annotations imported before
        db_query = self.dbConnector.execute(sql.SQL('''
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
            pl=sql.SQL(','.join(['%s' for _ in file_keys])),
            id_img=sql.Identifier(self.project, 'image'),
            id_iu=sql.Identifier(self.project, 'image_user')
        ),
        tuple((v,) for v in file_keys.values()), 'all')
        img_lut = defaultdict(list)
        for row in db_query:
            if verify_image_size or row['width'] is None or row['height'] is None:
                fname = os.path.join(self.projectRoot, row['filename'])
                driver = drivers.get_driver(fname)
                imsize = driver.size(fname)
                row['width'] = imsize[2]
                row['height'] = imsize[1]
            img_lut[row['labelname']].append(row)

        # filter valid images: found in database, with valid annotation, and not blacklisted
        for file in seg_files:
            if file not in file_keys or file_keys[file] not in img_lut:
                warnings.append(
                    f'Annotation file "{file}": no equivalent image found in AIDE project.'
                )
                continue

            # load map
            arr = drivers.load_from_disk(os.path.join(self.tempDir, file))

            # get unique classes across channel dimension
            num_bands = arr.shape[0]
            if num_bands == 1:
                # segmask is already index-based
                arr_out = np.copy(arr).squeeze().astype(np.uint8)

                # check for non-existing classes
                class_idx_new = lc_set_idx.difference(set(np.unique(arr)))
                labelclasses_new.add(class_idx_new)

                if skipUnknownClasses:
                    # set new classes to zero
                    for class_idx in class_idx_new:
                        arr_out[arr_out==class_idx] = 0
                else:
                    # check if new classes still allowed as per max idx ordinal serial
                    for class_idx in class_idx_new:
                        if class_idx not in labelclasses_new:
                            if self.max_idx >= 255:
                                # maximum reached
                                labelclasses_dropped.add(class_idx)
                                arr_out[arr_out==class_idx] = 0
                            else:
                                # still capacity; add
                                self.max_idx = max(self.max_idx, class_idx)

            else:
                # RGB; convert from colors
                arr_flat = np.reshape(arr, (arr.shape[0], -1))
                arr_colors = np.unique(arr_flat, axis=1)
                arr_colors = set(tuple(a) for a in arr_colors.T.tolist())

                arr_out = np.zeros(shape=arr.shape[1:3], dtype=np.uint8)

                # check for non-existing classes
                colors_new = set(lc_lut_color.keys()).difference(arr_colors)

                for rgb in arr_colors:
                    if skipUnknownClasses and rgb in colors_new:
                        # skip in-painting
                        continue
                    else:
                        if rgb not in lc_lut_color:
                            # new color; check if capacity available as per max idx ordinal serial
                            if self.max_idx >= 255:
                                # maximum reached
                                labelclasses_dropped.add(rgb)
                                continue

                            # still capacity; add
                            idx = self.max_idx + 1
                            labelclasses_new.add(idx)
                            lc_lut_color[rgb] = idx
                            self.max_idx += 1

                        idx = lc_lut_color[rgb]
                        valid = (arr[0,...] == rgb[0]) * \
                                (arr[1,...] == rgb[1]) * \
                                (arr[2,...] == rgb[2])
                        arr_out[valid] = idx

            if not np.sum(arr_out):
                warnings.append(
                    f'Annotation file "{file}": empty annotations or no valid label classes found.'
                )
                if skip_empty_images:
                    continue

            # register in database, accounting for virtual views
            imgs_insert = []
            img_entries = img_lut[file_keys[file]]
            for entry in img_entries:
                if entry.get('x', None) is not None and entry.get('y', None) is not None:
                    # virtual view; crop segmentation mask at given position
                    arr_crop = np.zeros(shape=(entry['width'], entry['height']), dtype=np.uint8)
                    bounds = (
                        min(arr_out.shape[0]-1, max(0, entry['x'])),        # left
                        min(arr_out.shape[1]-1, max(0, entry['y'])),        # top
                        min(arr_out.shape[0], entry['x']+entry['width']),   # right
                        min(arr_out.shape[1], entry['y']+entry['height']),  # bottom
                    )
                    arr_crop[:bounds[2]-bounds[0],:bounds[3]-bounds[1]] = \
                                    arr_out[bounds[0]:bounds[2], bounds[1]:bounds[3]]
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
            if len(imgs_insert) > 0:
                db_insert = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, timeCreated, timeRequired, unsure, segmentationMask, width, height)
                    VALUES %s
                    RETURNING image, id;
                ''').format(sql.Identifier(self.project, 'annotation')),
                tuple(imgs_insert), 'all')
                for row in db_insert:
                    imported_annotations[row[0]].append(row[1])

        # register new / update existing label classes
        if (not skipUnknownClasses and len(labelclasses_new)) > 0 or len(labelclasses_update) > 0:
            lc_colors = set(lc_lut_color.keys())
            lc_vals = []
            if not skipUnknownClasses:
                for label_class in labelclasses_new:
                    color = helpers.randomHexColor(lc_colors)
                    lc_colors.add(color)
                    lc_vals.append((f'Class {label_class}', label_class, color))

            # add color for existing classes to be updated
            for lc_id in labelclasses_update:
                color = helpers.randomHexColor(lc_colors)
                lc_colors.add(color)
                lc_name = self.labelClasses[lc_id]['name']
                lc_idx = self.labelClasses[lc_id]['idx']
                lc_vals.append((lc_name, lc_idx, color))
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (name, idx, color)
                VALUES %s
                ON CONFLICT (idx) DO
                UPDATE SET color = EXCLUDED.color;
            ''').format(sql.Identifier(self.project, 'labelclass')),
            tuple(lc_vals))

        if len(labelclasses_dropped) > 0:
            warnings.append('''The following label class indices and/or RGB colors were identified
                            but could not be added to the project: ''' + \
                            ', '.join(str(l) for l in labelclasses_dropped) + \
                            '.\nPixels annotated with those classes were set to zero.')

        # also set in image_user relation
        if len(imported_annotations) > 0:
            img_ids_added = list(imported_annotations.keys())
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (username, image, last_checked, last_time_required)
                VALUES %s
                ON CONFLICT (username, image) DO UPDATE
                SET last_time_required = -1;
            ''').format(sql.Identifier(self.project, 'image_user')),
            tuple((targetAccount, i, now, -1) for i in img_ids_added))

            if markAsGoldenQuestions:
                self.dbConnector.insert(sql.SQL('''
                    UPDATE {}
                    SET isGoldenQuestion = TRUE
                    WHERE id IN (%s);
                ''').format(sql.Identifier(self.project, 'image')),
                tuple((i,) for i in img_ids_added))

        return {
            'result': dict(imported_annotations),
            'warnings': warnings,
            'errors': errors
        }


    def export_annotations(self, annotations, destination, **kwargs):

        # if True, RGB images with labelclass colors will be exported instead of single channel
        # index maps
        export_colors = kwargs.get('export_colors', False)

        now = helpers.slugify(helpers.current_time())

        # export labelclass definition
        lc_str = 'labelclass,index,color\n'
        for label_class in self.labelClasses.values():
            lc_str += '{},{},{}\n'.format(
                label_class['name'], label_class['idx'], label_class['color']
            )
        destination.writestr('classes.txt', data=lc_str)

        # create labelclass color lookup table if needed
        if export_colors:
            color_lut = {}
            for lc_def in self.labelClasses.values():
                idx = lc_def['idx']
                try:
                    color = helpers.hexToRGB(lc_def['color'])
                except Exception:
                    # error parsing color; this should never happen
                    color = 0
                color_lut[idx] = color

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
        for filename, img_views in img_lut.items():
            # get total image file properties
            with rasterio.open(os.path.join(self.projectRoot, filename)) as src:
                size = [src.count, src.height, src.width]
                transform = src.transform
                crs = src.crs

            dest_files = {}      # username: file name

            for img_view in img_views:
                img_id = img_view['id']
                fname, _ = os.path.splitext(filename)

                for anno in anno_lut[img_id]:
                    # put potentially multiple segmentation masks into different subfolders
                    username = str(anno.get('username', 'cnnstate'))
                    if username not in dest_files:
                        dest_filename = os.path.join('segmentation', username, fname+'.tif')
                        dest_files[username] = dest_filename
                        temp_dest = os.path.join(self.tempDir, 'segmentation_export', now,
                                                dest_filename)
                        parent, _ = os.path.split(temp_dest)
                        os.makedirs(parent, exist_ok=True)
                    else:
                        temp_dest = os.path.join(self.tempDir, 'segmentation_export', now,
                                                dest_files[username])

                    # load segmap
                    segmap = self.dbConnector.execute(sql.SQL('''
                        SELECT segmentationMask
                        FROM {}
                        WHERE id = %s;
                    ''').format(sql.Identifier(self.project, annotations['data_type'])),
                    (anno['id'],), 1)

                    # convert base64 mask to image
                    raster_raw = np.frombuffer(base64.b64decode(segmap[0]['segmentationmask']),
                                                dtype=rasterio.ubyte)
                    raster_raw = np.reshape(raster_raw,
                                            (1, int(anno['height']),int(anno['width']),))

                    # convert to RGB image if needed
                    if export_colors:
                        raster_raw = raster_raw.squeeze()
                        raster = np.zeros((3, *raster_raw.shape[:2]), dtype=rasterio.ubyte)
                        lc_indices = np.unique(raster_raw).tolist()
                        for lc_idx in lc_indices:
                            if lc_idx not in color_lut:
                                continue
                            val = color_lut[lc_idx]
                            valid = (raster_raw == lc_idx)
                            for value in range(3):
                                raster[value,valid] = val[value]
                    else:
                        raster = raster_raw

                    # window where to write segmask part into
                    window = Window(
                        img_view['x'],
                        img_view['y'],
                        img_view['width'],
                        img_view['height']
                    )
                    with rasterio.open(
                        temp_dest, 'w',
                        driver='GTiff',
                        nodata=0,
                        width=size[2], height=size[1], count=3 if export_colors else 1,
                        transform=transform, crs=crs,
                        dtype=rasterio.ubyte) as dst:
                        dst.write(raster, window=window, indexes=list(range(1,len(raster)+1)))

            # move all temp files into zipfile
            for dest_filename in dest_files.values():
                temp_dest = os.path.join(self.tempDir, 'segmentation_export', now, dest_filename)
                destination.write(temp_dest, dest_filename)
                os.remove(temp_dest)
                files_exported.append(dest_filename)

        shutil.rmtree(os.path.join(self.tempDir, 'segmentation_export', now))

        return {
            'files': tuple(files_exported)
        }
