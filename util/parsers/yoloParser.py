'''
    Label parser for annotations in YOLO format:
    https://pjreddie.com/darknet/yolo/

    2022-23 Benjamin Kellenberger
'''

import os
from collections import defaultdict
import re
import difflib
import yaml
from psycopg2 import sql

from util.parsers.abstractParser import AbstractAnnotationParser
from util import helpers, drivers
drivers.init_drivers()


class YOLOparser(AbstractAnnotationParser):

    NAME = 'YOLO'
    INFO = '<p>Supports annotations of labels, bounding boxes, and polygons in the YOLO format.'
    ANNOTATION_TYPES = ('boundingBoxes',)

    FILE_SUB_PATTERN = '(\/|\\\\)*.*\/*(images|labels|annotations)(\/|\\\\)*'           # pattern that attempts to identify image file name from label file name


    '''
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, images with zero annotations will not be added to the "image_user" relation
        parseYOLOv5MetaFiles = kwargs.get('parse_yolov5_meta_files', False)
    '''

    @classmethod
    def get_html_options(cls, method):
        if method == 'import':
            return '''
            <div>
                <input type="checkbox" id="skip_empty_images" />
                <label for="skip_empty_images">skip images without annotations</label>
            </div>
            '''
        else:
            return ''


    @classmethod
    def _get_yolo_files(cls, fileDict, folderPrefix=''):
        '''
            Iterates through a provided dict of file names (original: actual)
            and returns all those that appear to be in YOLO format:
            - text files, one per image
            - file name corresponds to image name (apart from top-level folder)
            - contents are lines with five white space-separated values:
                <class idx> <x> <y> <width> <height>
              <class idx> is an int, the rest are floats in [0, 1]
        '''
        yoloFiles = {
            'annotation': {},
            'meta': []              # meta files, such as YAML-formatted definitions of datasets as in YOLOv5 (example: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)
        }
        for file_orig, file_new in fileDict.items():
            if isinstance(folderPrefix, str):
                filePath = os.path.join(folderPrefix, file_orig)
            else:
                filePath = file_orig
            if not os.path.isfile(filePath) and not os.path.islink(filePath):
                continue
            if helpers.is_binary(filePath):
                continue

            # check if original YOLO classes.txt file
            _, file_name = os.path.split(filePath)
            if file_name.lower() == 'classes.txt':
                yoloFiles['meta'].append(filePath)

            # check if YOLOv5 YAML file
            try:
                with open(filePath, 'r', encoding='utf-8') as f_yaml:
                    meta = yaml.safe_load(f_yaml)
                assert isinstance(meta, dict)
                assert 'nc' in meta     # number of classes
                assert 'names' in meta  # labelclass names

                # valid YOLOv5 YAML file
                yoloFiles['meta'].append(filePath)
                continue
            except Exception:
                # no YAML file; try next
                pass

            try:
                with open(filePath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    tokens = line.strip().split(' ')
                    assert len(tokens) == 5
                    assert isinstance(int(tokens[0]), int)                           # label class
                    assert all(isinstance(float(t), float) for t in tokens[1:])      # bbox

                # no errors encountered; this is a YOLO file
                yoloFiles['annotation'][file_orig] = file_new

            except Exception:
                # unparseable or erroneous format
                continue

        return yoloFiles


    @classmethod
    def is_parseable(cls, fileDict, folderPrefix=''):
        '''
            File list is parseable if at least one file is a proper YOLO
            annotation file.
        '''
        return len(cls._get_yolo_files(fileDict, folderPrefix)['annotation']) > 0     #TODO: we could parse the YAML file and download the data from there, but this requires importing images, too


    def import_annotations(self, fileDict, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):

        # args setup
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, images with zero annotations will not be added to the "image_user" relation

        now = helpers.current_time()

        importedAnnotations, warnings, errors = defaultdict(list), [], []
        imgIDs_added = set()

        uploadedImages = set([f for f in fileDict.keys() if fileDict[f] != '-1'])

        # get valid YOLO files
        yoloFiles = self._get_yolo_files(fileDict, self.tempDir)
        if not len(yoloFiles['annotation']):
            return {
                'result': {},
                'warnings': [],
                'errors': ['No valid YOLO annotation file found.']
            }

        labelclasses = {}

        # check if YOLOv5 YAML or YOLO classes.txt file provided
        if len(yoloFiles['meta']) > 0:
            for meta_file in yoloFiles['meta']:
                _, file_name = os.path.split(meta_file)
                if file_name.lower() == 'classes.txt':
                    # original YOLO classes file
                    with open(meta_file, 'r', encoding='utf-8') as f_classes:
                        lines = f_classes.readlines()
                        labelclasses = dict(zip(range(len(lines)),
                                                [line.strip() for line in lines]))
                else:
                    # YOLOv5 YAML file
                    with open(meta_file, 'r', encoding='utf-8') as f_meta:
                        meta = yaml.safe_load(f_meta)
                    for idx, labelclass in enumerate(meta['names']):
                        if labelclass not in self.labelClasses and skipUnknownClasses:
                            continue
                        labelclasses[idx] = labelclass

        # iterate over identified YOLO files
        meta = {}
        for fpath in yoloFiles['annotation']:
            # file key to match the annotation file with the image(s) present in the database
            if fpath.startswith(os.sep) or fpath.startswith('/'):
                fpath = fpath[1:]

            fbody = re.sub(self.FILE_SUB_PATTERN, '', fpath, flags=re.IGNORECASE)        # replace "<base folder>/(labels|annotations)/*" with "%/*" for search in database with "LIKE" expression
            fbody = os.path.splitext(fbody)[0]

            # check if YOLO anno file name can be matched with an image uploaded in the same session
            if len(uploadedImages) > 0:
                candidates = difflib.get_close_matches(fbody, uploadedImages, n=1)
                if len(candidates):
                    fKey = fileDict[candidates[0]]      # use new, potentially overwritten file name
                else:
                    # no exact match; check database by using fbody directly
                    fKey = fbody
            else:
                # only annotations uploaded; check database by using fbody directly
                fKey = fbody

            # load bboxes
            labels, bboxes = [], []

            # load newly saved file
            with open(os.path.join(self.tempDir, fpath), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) == 0:
                warnings.append(
                    f'Annotation file "{fpath}": no annotations registered.'
                )
                if skipEmptyImages:
                    continue
            has_skipped_annotations = False
            for line in lines:
                tokens = line.strip().split(' ')
                lc = int(tokens[0])
                if lc not in labelclasses:
                    if skipUnknownClasses:
                        has_skipped_annotations = True
                        continue
                    labelclasses[lc] = f'Class {lc}'        # we cannot infer labelclass names any other way (TODO: check for txt file at base of directory, cf. LabelImg)
                labels.append(lc)
                bboxes.append([float(t) for t in tokens[1:]])      #TODO: verification of validity of annotations?

            if has_skipped_annotations:
                if len(labels) == 0 and skipEmptyImages:
                    warnings.append(
                        f'Annotation file "{fpath}": all label classes unknown; file skipped.'
                    )
                    continue
                warnings.append(
                    f'Annotation file "{fpath}": ' + \
                        'unknown label classes encountered; affected annotations skipped.'
                )

            meta[fKey] = {
                'file': fpath,
                'labels': labels,
                'bboxes': bboxes
            }

        if len(meta) == 0:
            return {
                'result': {},
                'warnings': warnings,
                'errors': ['No valid YOLO annotation file found.']
            }

        # find corresponding images in database, blacklisting those that had annotations imported before
        dbQuery = self.dbConnector.execute(sql.SQL('''
            WITH labelquery(labelname) AS (
                VALUES {pl}
            )
            SELECT id, filename, x, y, width, height, labelname
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
            pl=sql.SQL(','.join(['%s' for _ in meta.keys()])),
            id_img=sql.Identifier(self.project, 'image'),
            id_iu=sql.Identifier(self.project, 'image_user')
        ),
        tuple((m,) for m in meta.keys()), 'all')

        # create image lookup table: also factor in multiple virtual views of same image
        img_lut = defaultdict(list)
        imgSizes = {}
        for d in dbQuery:
            # get original image size (pre-virtual views); this also checks for image readability
            filename = d['filename']
            if filename not in imgSizes:
                try:
                    file_path = os.path.join(self.projectRoot, filename)
                    driver = drivers.get_driver(file_path)
                    img_size = driver.size(file_path)
                    imgSizes[filename] = img_size
                except Exception:
                    # image is not loadable     #TODO: mark as corrupt in database
                    warnings.append(
                        f'Image "{filename}" could not be loaded; affected annotations skipped.'
                    )
                    imgSizes[filename] = None

            imgSize = imgSizes[filename]
            if imgSize is None:
                # unloadable image; skip annotation
                continue

            img_lut[d['labelname']].append({
                'id': d['id'],
                'window': [d['y'], d['x'], d['height'], d['width']],
                'full_img_size': imgSize
            })

        # register label classes
        num_annotations_skipped = 0
        if not skipUnknownClasses:
            lcs_new = set(labelclasses.values()).difference(set(self.labelClasses.keys()))
            if len(lcs_new) > 0:
                self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (name)
                    VALUES %s;
                ''').format(sql.Identifier(self.project, 'labelclass')),
                tuple((l,) for l in lcs_new))

                # update local cache
                self._init_labelclasses()

        # substitute YOLO labelclass numbers with UUIDs and append image ID
        imgIDs_added = set()
        for fkey, annoData in meta.items():
            insertVals = []
            if fkey not in img_lut:
                num_annotations_skipped += len(annoData['labels'])
                continue
            imgs = img_lut[fkey]
            for l in range(len(annoData['labels'])):
                label = labelclasses[annoData['labels'][l]]
                if label not in self.labelClasses:
                    num_annotations_skipped += 1
                    continue
                label = self.labelClasses[label]
                bbox = annoData['bboxes'][l]

                # find matching target image entry
                if len(imgs) == 1:
                    # only one image (no virtual views)
                    imgID = imgs[0]['id']
                else:
                    # virtual views generated; find the one encompassing the annotation
                    fullImageSize = imgs[0]['full_img_size']
                    bbox_abs = [
                        bbox[0]*fullImageSize[2],
                        bbox[1]*fullImageSize[1],
                        bbox[2]*fullImageSize[2],
                        bbox[3]*fullImageSize[1]
                    ]
                    bbox_abs[0] -= bbox_abs[2]/2.0
                    bbox_abs[1] -= bbox_abs[3]/2.0
                    for imgView in imgs:
                        imgWindow = imgView['window']
                        if bbox_abs[0] < imgWindow[1]+imgWindow[3] and \
                            bbox_abs[1] < imgWindow[0]+imgWindow[2] and \
                                bbox_abs[0]+bbox_abs[2] > imgWindow[1] and \
                                    bbox_abs[1]+bbox_abs[3] > imgWindow[0]:
                            # correct virtual view found; adjust bbox           #TODO: currently no cropping, only shifting
                            bbox = [
                                (bbox_abs[0] - imgWindow[1] + bbox_abs[2]/2.0) / imgWindow[3],
                                (bbox_abs[1] - imgWindow[0] + bbox_abs[3]/2.0) / imgWindow[2],
                                bbox_abs[2] / imgWindow[3],
                                bbox_abs[3] / imgWindow[2]
                            ]
                            imgID = imgView['id']
                            break

                insertVals.append((
                    targetAccount, imgID, now, -1, False,
                    label, *bbox
                ))
                imgIDs_added.add(imgID)

            # add annotations to database
            if len(insertVals) > 0:
                result = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, timeCreated, timeRequired,
                                    unsure, label, x, y, width, height)
                    VALUES %s
                    RETURNING image, id;
                ''').format(sql.Identifier(self.project, 'annotation')),
                tuple(insertVals),
                'all')
                for row in result:
                    importedAnnotations[row[0]].append(row[1])

        # also set in image_user relation
        if len(imgIDs_added) > 0:
            imgIDs_added = list(imgIDs_added)
            self.dbConnector.insert(sql.SQL('''
                INSERT INTO {} (username, image, last_checked, last_time_required)
                VALUES %s
                ON CONFLICT (username, image) DO UPDATE
                SET last_time_required = -1;
            ''').format(sql.Identifier(self.project, 'image_user')),
            tuple((targetAccount, i, now, -1) for i in imgIDs_added))

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

        # create new labelclass index for YOLO format       #TODO: sort after labelclass name first?
        categoryLookup = dict([annotations['labelclasses'][l]['id'], l]
                              for l in range(len(annotations['labelclasses'])))

        # query image window sizes and create LUT
        imgSizeLookup = {}
        imgLookup = {}
        for image in annotations['images']:
            filename = image['filename']
            window = [image['y'], image['x'], image['height'], image['width']]
            if all(w is not None for w in window) and filename not in imgSizeLookup:
                # virtual window: get original image size
                file_path = os.path.join(self.projectRoot, filename)
                driver = drivers.get_driver(file_path)
                sz = driver.size(file_path)
                imgSizeLookup[filename] = [float(s) for s in sz]

            imgLookup[image['id']] = {
                'filename': filename,
                'window': window
            }

        # export annotations
        export = {}
        for anno in annotations['annotations']:
            img_id = anno['image']
            if img_id not in export:
                export[img_id] = {
                    'labels': [],
                    'bboxes': []
                }
            export[img_id]['labels'].append(categoryLookup[anno['label']])

            # check if bbox is to be shifted
            bbox = [anno['x'], anno['y'], anno['width'], anno['height']]
            window = imgLookup[img_id]['window']
            if all(w is not None for w in window):
                # virtual view; shift bbox
                full_img_size = imgSizeLookup[imgLookup[img_id]['filename']]
                bbox[0] = (bbox[0]*window[3] + window[1])/full_img_size[2]
                bbox[1] = (bbox[1]*window[2] + window[0])/full_img_size[1]
                bbox[2] *= window[3]/full_img_size[2]
                bbox[3] *= window[2]/full_img_size[1]
            export[img_id]['bboxes'].append(bbox)

        # export
        files_exported = []
        #TODO: info and license files
        # labelclass names
        lc_lut = dict((v,k) for k, v in self.labelClasses.items())
        destination.writestr('classes.txt',
            data='\n'.join([lc_lut[lcKey] for lcKey in categoryLookup.keys()]))
        files_exported.append('classes.txt')

        # consolidate to one file per image (accounting for potential virtual views)
        outputs = defaultdict(list)

        for img_id, entry in export.items():
            img_filename = imgLookup[img_id]['filename']
            base_filename, _ = os.path.splitext(img_filename)
            label_filename = base_filename + '.txt'

            labels, bboxes = entry['labels'], entry['bboxes']
            label_file_contents = '\n'.join([
                '{} {} {} {} {}'.format(
                    labels[l],
                    *bboxes[l]
                )
            for l in range(len(labels))])
            outputs[label_filename].append(label_file_contents)

        # write consolidated outputs
        for file_name, contents in outputs.items():
            destination.writestr(file_name, data='\n'.join(contents))
            files_exported.append(file_name)

        return {
            'files': tuple(files_exported)
        }



#TODO
if __name__ == '__main__':

    project = 'bboxes'
    fileDir = '/data/datasets/Kuzikus/patches_800x600'      #TODO
    targetAccount = 'bkellenb'
    annotationType = 'boundingBoxes'

    from tqdm import tqdm
    import glob
    fileList = glob.glob(os.path.join(fileDir, '**/*'), recursive=True)


    from util.configDef import Config
    from modules.Database.app import Database

    config = Config()
    dbConnector = Database(config)


    if False:
        # annotation import

        # import images first: cheap way out by linking images into project
        if not fileDir.endswith(os.sep):
            fileDir += os.sep
        projectDir = os.path.join(config.getProperty('FileServer', 'staticfiles_dir'), project)
        for fname in tqdm(fileList):
            if not os.path.isfile(fname) or not fname.lower().endswith('.jpg'):
                continue
            fname_dest = os.path.join(projectDir, fname.replace(fileDir, ''))
            if os.path.exists(fname_dest):
                continue
            parent, _ = os.path.split(fname_dest)
            os.makedirs(parent, exist_ok=True)
            os.symlink(fname, fname_dest)
        
        from modules.DataAdministration.backend.dataWorker import DataWorker
        dw = DataWorker(config, dbConnector)
        dw.addExistingImages(project, skipIntegrityCheck=True)

        # then, parse and import annotations
        parser = YOLOparser(config, dbConnector, project, targetAccount, annotationType)

        kwargs = {
            'skip_empty_images': True
        }

        result = parser.import_annotations(fileList, targetAccount, skipUnknownClasses=False, markAsGoldenQuestions=True, **kwargs)
    

    else:
        # annotation export
        from modules.DataAdministration.backend.dataWorker import DataWorker
        dw = DataWorker(config, dbConnector)
        outFile = dw.requestAnnotations(project, targetAccount, 'yolo', 'annotation', ignoreImported=False)

        print(outFile)