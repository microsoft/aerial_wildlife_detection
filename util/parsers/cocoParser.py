'''
    Label parser for geometric annotations in MS-COCO format:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from psycopg2 import sql

from util.parsers.abstractParser import AbstractAnnotationParser
from util import helpers, drivers


class COCOparser(AbstractAnnotationParser):

    ANNOTATION_TYPES = ('labels', 'points', 'boundingBoxes')

    def __init__(self, config, dbConnector, project):
        super(COCOparser, self).__init__(config, dbConnector, project)

        # get project annotation and prediction types
        labelTypes = self.dbConnector.execute(sql.SQL('''
            SELECT annotationType, predictionType
            FROM "aide_admin".project
            WHERE shortname = %s;
        '''), (self.project,), 1)
        labelTypes = {
            'annotations': labelTypes[0]['annotationtype'],
            'predictions': labelTypes[0]['predictiontype']
        }


    def _get_coco_files(self, fileList):
        '''
            Iterates through a provided list of file names and returns all those
            that appear to be valid MS-COCO JSON files.
        '''
        cocoFiles = []
        for f in fileList:
            if not os.path.isfile(f) and not os.path.islink(f):
                continue
            _, ext = os.path.splitext(f.lower())
            if not ext in ('.json', '.txt', ''):
                continue
            try:
                meta = json.load(open(f, 'r'))

                # valid JSON file; check for required fields
                assert 'images' in meta
                assert isinstance(meta['images'], list) or isinstance(meta['images'], tuple)
                assert 'categories' in meta
                assert isinstance(meta['categories'], list) or isinstance(meta['categories'], tuple)
                assert 'annotations' in meta
                assert isinstance(meta['annotations'], list) or isinstance(meta['annotations'], tuple)

                # basic supercategories are present; we assume this is indeed a valid COCO file
                cocoFiles.append(f)

            except:
                # unparseable or erroneous format
                continue
        
        return cocoFiles


    def is_parseable(self, fileList):
        '''
            File list is parseable if at least one file is a proper MS-COCO JSON
            file.
        '''
        return len(self._get_coco_files(fileList)) > 0


    def import_annotations(self, fileList, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):

        # args setup
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, images with zero annotations will not be added to the "image_user" relation
        verifyImageSize = kwargs.get('verify_image_size', False)    # if True, image sizes will be retrieved from files, not just from COCO metadata
        unsure = kwargs.get('mark_iscrowd_as_unsure', False)        # if True, annotations with attribute "iscrowd" will be marked as "unsure" in AIDE

        now = helpers.current_time()

        ids, warnings, errors = [], [], []
        imgIDs_added = set()
        imgs_valid = []

        # get valid COCO files
        cocoFiles = self._get_coco_files(fileList)
        if not len(cocoFiles):
            return {
                'ids': [],
                'warnings': [],
                'errors': ['No valid MS-COCO annotation file found.']
            }
        
        # iterate over identified COCO files
        for cf in cocoFiles:
            meta = json.load(open(cf, 'r'))

            # gather label classes. Format: COCO id: (name, supercategory (if present))
            labelClasses = dict(zip([c['id'] for c in meta['categories']],
                                [(c['name'], c.get('supercategory', None)) for c in meta['categories']]))
            
            if skipUnknownClasses:
                lcNames = set([l[0] for l in labelClasses.values()])
                lcIntersection = set(self.labelClasses.keys()).intersection(lcNames)
                if not len(lcIntersection):
                    errors.append(f'"{cf}": no common label classes with project found.')
                    continue

            # assemble image file LUT
            images = dict(zip([i['file_name'] for i in meta['images']], [i for i in meta['images']]))

            # assemble annotation LUT
            annotations = {}
            for imgKey in images.values():
                annotations[imgKey['id']] = []
            for anno in meta['annotations']:
                try:
                    if anno['image_id'] in annotations:
                        assert all([a > 0.0 for a in anno['bbox']])
                        annotations[anno['image_id']].append(anno)
                except:
                    # corrupt annotation; ignore
                    pass

            # find intersection with project images
            imgs_match = self.match_filenames(list(images.keys()))

            # blacklist images where annotations had already been imported before (identifiable by a negative "timeRequired" attribute)
            imgs_blacklisted = self.dbConnector.execute(sql.SQL('''
                SELECT image
                FROM {}
                WHERE timeRequired < 0
            ''').format(sql.Identifier(self.project, 'annotation')), None, 'all')
            imgs_blacklisted = set([i['image'] for i in imgs_blacklisted])

            # filter valid images: found in database, with valid annotation, and not blacklisted
            for img in imgs_match:
                imgUUID, imgPath = img[0], img[1]
                if imgUUID in imgs_blacklisted:
                    warnings.append(
                        f'Annotations for image with ID {imgCOCOid} ({imgPath}) had already been imported before and are skipped in this import.'
                    )
                    continue

                imgCOCOmeta = images[imgPath]
                imgCOCOid = imgCOCOmeta['id']
                if img is None:
                    warnings.append(
                        f'Image with ID {imgCOCOid} ({imgPath}) could not be found in project.'
                    )
                    continue
                
                # get corresponding annotations
                anno = annotations.get(imgCOCOid, None)
                if anno is None or not len(anno):
                    warnings.append(
                        f'Image with ID {imgCOCOid} ({imgPath}): no annotations registered.'
                    )
                    if skipEmptyImages:
                        continue

                if skipUnknownClasses:
                    # check for validity of label classes
                    anno = [a for a in anno if a['category_id'] in labelClasses]
                    if not len(anno):
                        warnings.append(
                            f'"Image with ID {imgCOCOid} ({imgPath}): annotations contain label classes that are not registered in project.'
                        )
                        continue
                
                annos_valid = []
                if len(anno):
                    # retrieve image dimensions
                    try:
                        if verifyImageSize or 'width' not in imgCOCOmeta or 'height' not in imgCOCOmeta:
                            imageSize = drivers.load_from_disk(os.path.join(self.projectRoot, imgPath)).shape[1:]
                        else:
                            imageSize = (imgCOCOmeta['height'], imgCOCOmeta['width'])
                    except:
                        # image could not be loaded
                        errors.append(
                            f'"Image with ID {imgCOCOid} ({imgPath}) could not be loaded.'
                        )
                        continue

                    # scale annotations to relative [0,1) coordinates
                    for a in range(len(anno)):
                        try:
                            bbox = [float(v) for v in anno[a]['bbox']]
                            assert len(bbox) == 4, f'invalid number of bounding box coordinates ({len(bbox)} != 4)'

                            # make relative
                            bbox[0] /= imageSize[1]
                            bbox[1] /= imageSize[0]
                            bbox[2] /= imageSize[1]
                            bbox[3] /= imageSize[0]
                            bbox[0] = min(1.0, bbox[0]+bbox[2]/2.0)
                            bbox[1] = min(1.0, bbox[1]+bbox[3]/2.0)

                            assert all([b > 0 and b <= 1.0 for b in bbox]), 'Invalid bounding box coordinates ({})'.format(bbox)

                            # everything alright so far
                            annos_valid.append([bbox, anno[a]['category_id']])
                        except Exception as e:
                            warnings.append(
                                f'Annotation {a} for image with ID {imgCOCOid} ({imgPath}) contained an error (reason: "{str(e)}" and was therefore skipped.'
                            )
                if len(annos_valid) or not skipEmptyImages:
                    imgs_valid.append({'img': imgUUID, 'anno': annos_valid})

            # add new label classes
            if not skipUnknownClasses:
                # add new supercategories first (if present)
                supercategories = [l[1] for l in labelClasses.values() if l[1] is not None]
                labelclassGroups = self.dbConnector.execute(sql.SQL('''
                    SELECT id, name
                    FROM {};
                ''').format(sql.Identifier(self.project, 'labelclassgroup')),
                None, 'all')
                labelclassGroups = dict([[l['name'], l['id']] for l in labelclassGroups])

                groups_new = set(supercategories).difference(set(labelclassGroups.keys()))
                if len(groups_new):
                    result = self.dbConnector.insert(sql.SQL('''
                        INSERT INTO {} (name)
                        VALUES %s
                        RETURNING id, name;
                    ''').format(sql.Identifier(self.project, 'labelclassgroup')),
                    [(s,) for s in supercategories], 'all')
                    # augment LUT
                    for r in result:
                        labelclassGroups[r[1]] = r[0]

                # add new label classes
                lcNames = set([l[0] for l in labelClasses.values()])
                lcDifference = lcNames.difference(set(self.labelClasses.keys()))
                if len(lcDifference):
                    self.dbConnector.insert(sql.SQL('''
                        INSERT INTO {} (name, labelclassgroup)
                        VALUES %s;
                    ''').format(sql.Identifier(self.project, 'labelclass')),
                    [(l,labelclassGroups.get(l, None)) for l in lcDifference])

                    # update LUT
                    self._init_labelclasses()
            
        if len(imgs_valid):
            for data in imgs_valid:
                if not len(data['anno']):
                    continue
                # replace label class names with IDs
                for a in range(len(data['anno'])):
                    catID = data['anno'][a][1]
                    lcName = labelClasses[catID][0]
                    data['anno'][a][1] = self.labelClasses[lcName]
            
                # finally, add annotations to database
                result = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, timeCreated, timeRequired, label, x, y, width, height, unsure)
                    VALUES %s
                    RETURNING id;
                ''').format(sql.Identifier(self.project, 'annotation')),
                tuple([(targetAccount, data['img'], now, -1,
                    anno[1],
                    anno[0][0], anno[0][1],
                    anno[0][2], anno[0][3], unsure) for anno in data['anno']]), 'all')
                ids.extend([r[0] for r in result])

            # also set in image_user relation
            imgIDs = [i['img'] for i in imgs_valid]
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
            'ids': ids,
            'warnings': warnings,
            'errors': errors
        }




#TODO
if __name__ == '__main__':

    project = 'cocotest'
    fileDir = '/data/datasets/Kuzikus/patches_800x600'
    targetAccount = 'bkellenb'

    from tqdm import tqdm
    import glob
    fileList = glob.glob(os.path.join(fileDir, '**/*'), recursive=True)


    from util.configDef import Config
    from modules.Database.app import Database

    config = Config()
    dbConnector = Database(config)

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
    parser = COCOparser(config, dbConnector, project)

    kwargs = {
        'skip_empty_images': True
    }

    result = parser.import_annotations(fileList, targetAccount, skipUnknownClasses=False, markAsGoldenQuestions=True, **kwargs)