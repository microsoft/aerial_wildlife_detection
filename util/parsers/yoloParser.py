'''
    Label parser for annotations in YOLO format:
    https://pjreddie.com/darknet/yolo/

    2022 Benjamin Kellenberger
'''

import os
from collections import defaultdict
import re
import yaml
from psycopg2 import sql

from util.parsers.abstractParser import AbstractAnnotationParser
from util import helpers


class YOLOparser(AbstractAnnotationParser):

    NAME = 'YOLO'
    INFO = '<p>Supports annotations of labels, bounding boxes, and polygons in the YOLO format.'
    ANNOTATION_TYPES = ('boundingBoxes',)

    FILE_SUB_PATTERN = '(\/|\\\\)*.*\/*(labels|annotations)(\/|\\\\)*'           # pattern that attempts to identify image file name from label file name


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
    def _get_yolo_files(cls, fileList):
        '''
            Iterates through a provided list of file names and returns all those
            that appear to be in YOLO format:
            - text files, one per image
            - file name corresponds to image name (apart from top-level folder)
            - contents are lines with five white space-separated values:
                <class idx> <x> <y> <width> <height>
              <class idx> is an int, the rest are floats in [0, 1]
        '''
        yoloFiles = {
            'annotation': [],
            'meta': []              # meta files, such as YAML-formatted definitions of datasets as in YOLOv5 (example: https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)
        }
        for file in fileList:
            if not os.path.isfile(file) and not os.path.islink(file):
                continue
            if helpers.is_binary(file):
                continue

            # check if YOLOv5 YAML file
            try:
                meta = yaml.safe_load(open(file, 'r'))
                assert isinstance(meta, dict)
                assert 'nc' in meta     # number of classes
                assert 'names' in meta  # labelclass names

                # valid YOLOv5 YAML file
                yoloFiles['meta'].append(file)
                continue
            except:
                # no YAML file; try next
                pass

            try:
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    tokens = line.strip().split(' ')
                    assert len(tokens) == 5
                    assert isinstance(int(tokens[0]), int)                           # label class
                    assert all([isinstance(float(t), float) for t in tokens[1:]])      # bbox

                # no errors encountered; this is a YOLO file
                yoloFiles['annotation'].append(file)

            except:
                # unparseable or erroneous format
                continue
            
        return yoloFiles


    @classmethod
    def is_parseable(cls, fileList):
        '''
            File list is parseable if at least one file is a proper YOLO
            annotation file.
        '''
        return len(cls._get_yolo_files(fileList)['annotation']) > 0     #TODO: we could parse the YAML file and download the data from there, but this requires importing images, too


    def import_annotations(self, fileList, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):

        # args setup
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, images with zero annotations will not be added to the "image_user" relation

        now = helpers.current_time()

        importedAnnotations, warnings, errors = defaultdict(list), [], []
        imgIDs_added = set()

        # get valid YOLO files
        yoloFiles = self._get_yolo_files(fileList)
        if not len(yoloFiles['annotation']):
            return {
                'result': {},
                'warnings': [],
                'errors': ['No valid YOLO annotation file found.']
            }
        
        labelclasses = {}

        # check if YOLOv5 YAML file provided
        if len(yoloFiles['meta']):
            for metaFile in yoloFiles['meta']:
                meta = yaml.safe_load(metaFile)
                for idx, labelclass in enumerate(meta['names']):
                    if labelclass not in self.labelclasses and skipUnknownClasses:
                        continue
                    labelclasses[idx] = labelclass
        
        # iterate over identified YOLO files
        meta = {}
        for file in yoloFiles['annotation']:
            # file key to match the annotation file with the image(s) present in the database
            fpath = file.replace(self.tempDir, '')       #TODO
            if fpath.startswith(os.sep) or fpath.startswith('/'):
                fpath = fpath[1:]
            fKey = re.sub(self.FILE_SUB_PATTERN, '', fpath, flags=re.IGNORECASE)        # replace "<base folder>/(labels|annotations)/*" with "%/*" for search in database with "LIKE" expression

            # load bboxes
            labels, bboxes = [], []
            with open(file, 'r') as f:
                lines = f.readlines()
            if not len(lines):
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
                if not len(labels) and skipEmptyImages:
                    warnings.append(
                        f'Annotation file "{fpath}": all label classes unknown; file skipped.'
                    )
                    continue
                warnings.append(
                    f'Annotation file "{fpath}": unknown label classes encountered; affected annotations skipped.'
                )

            meta[fKey] = {
                'file': fpath,
                'labels': labels,
                'bboxes': bboxes
            }

        if not len(meta):
            return {
                'result': {},
                'warnings': warnings,
                'errors': ['No valid YOLO annotation file found.']
            }

        # find corresponding images in database, blacklisting those that had annotations imported before
        dbQuery = self.dbConnector.execute(sql.SQL('''
            WITH labelquery(labelname) AS (
                VALUES %s
            )
            SELECT id, filename, labelname
            FROM (
                SELECT id, filename
                FROM {id_img}
                WHERE id NOT IN (
                    SELECT image
                    FROM {id_iu}
                    WHERE last_time_required < 0
                )
            ) AS q
            CROSS JOIN labelquery
            WHERE filename LIKE CONCAT('%%', labelname);
        ''').format(
            id_img=sql.Identifier(self.project, 'image'),
            id_iu=sql.Identifier(self.project, 'image_user')
        ),
        (tuple([tuple([(l,) for l in meta.keys()])]),), 'all')

        img_lut = dict([[d['labelname'], d['id']] for d in dbQuery])

        # register label classes
        if not skipUnknownClasses:
            lcs_new = set(labelclasses.keys()).difference(set(self.labelClasses.keys()))
            if len(lcs_new):
                self.dbConnector.execute(sql.SQL('''
                    INSERT INTO {} (name)
                    VALUES %s;
                ''').format(sql.Identifier(self.project, 'labelclass')),
                tuple([(l,) for l in lcs_new]))

                # update local cache
                self._init_labelclasses()
        
        # substitute YOLO labelclass numbers with UUIDs and append image ID
        for fkey in meta.keys():
            imgID = img_lut[fkey]
            annoData = meta[fkey]
            for l in range(len(annoData['labels'])):
                annoData['labels'][l] = self.labelClasses[annoData['labels'][l]]

            # add annotations to database
            if len(annoData['labels']):
                result = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, timeCreated, timeRequired, unsure, label, x, y, width, height)
                    VALUES %s
                    RETURNING image, id;
                ''').format(sql.Identifier(self.project, 'annotation')),
                tuple([(targetAccount, imgID, now, -1, False,
                    annoData['labels'][l], *annoData['bboxes'][l]) for l in range(len(annoData['labels']))]),
                'all')
                for row in result:
                    importedAnnotations[row[0]].append(row[1])

        # also set in image_user relation
        imgIDs = [img_lut[i] for i in meta.keys()]
        self.dbConnector.insert(sql.SQL('''
            INSERT INTO {} (username, image, last_checked, last_time_required)
            VALUES %s
            ON CONFLICT (username, image) DO UPDATE
            SET last_time_required = -1;
        ''').format(sql.Identifier(self.project, 'image_user')),
        tuple([(targetAccount, i, now, -1) for i in imgIDs]))
        imgIDs_added = set(imgIDs)

        if markAsGoldenQuestions and len(imgIDs_added):
            self.dbConnector.insert(sql.SQL('''
                UPDATE {}
                SET isGoldenQuestion = TRUE
                WHERE id IN (%s);
            ''').format(sql.Identifier(self.project, 'image')),
            tuple([(i,) for i in list(imgIDs_added)]))

        return {
            'result': importedAnnotations,
            'warnings': warnings,
            'errors': errors
        }


    def export_annotations(self, annotations, destination, **kwargs):

        # create new labelclass index for YOLO format       #TODO: sort after labelclass name first?
        categoryLookup = dict([[annotations['labelclasses'][l]['id'], l] for l in range(len(annotations['labelclasses']))])
        
        # create image LUT
        imgIDLookup = dict([[i['id'], i['filename']] for i in annotations['images']])
     
        # export annotations
        export = {}
        for anno in annotations['annotations']:
            imgID = anno['image']
            if imgID not in export:
                export[imgID] = {
                    'labels': [],
                    'bboxes': []
                }
            export[imgID]['labels'].append(categoryLookup[anno['label']])
            export[imgID]['bboxes'].append([anno['x'], anno['y'], anno['width'], anno['height']])
        
        # export
        files_exported = []
        #TODO: info and license files
        # labelclass names
        lc_lut = dict([(v,k) for k, v in self.labelClasses.items()])
        destination.writestr('classes.txt',
            data='\n'.join([lc_lut[lcKey] for lcKey in categoryLookup.keys()]))
        files_exported.append('classes.txt')

        for imgID in export.keys():
            imgFilename = imgIDLookup[imgID]
            baseFilename, _ = os.path.splitext(imgFilename)
            labelFilename = baseFilename + '.txt'

            labels, bboxes = export[imgID]['labels'], export[imgID]['bboxes']
            labelFileContents = '\n'.join([
                '{} {} {} {} {}'.format(
                    labels[l],
                    *bboxes[l]
                )
            for l in range(len(labels))])

            destination.writestr(labelFilename, data=labelFileContents)
            files_exported.append(labelFilename)

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