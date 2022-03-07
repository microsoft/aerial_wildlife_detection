'''
    Label parser for geometric annotations in MS-COCO format:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from psycopg2 import sql

from util.parsers.abstractParser import AbstractAnnotationParser
from util import drivers


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
        clipBoundingBoxes = kwargs.get('clip_boxes', False)     # limit bounding boxes to image dimensions

        warnings, errors = [], []

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
            images = dict(zip([i['id'] for i in meta['images']], [i['file_name'] for i in meta['images']]))

            # assemble annotation LUT
            annotations = {}
            for imgKey in images.keys():
                annotations[imgKey] = []
            for anno in meta['annotations']:
                try:
                    if anno['image_id'] in annotations:
                        annotations[anno['image_id']].append(anno)
                except:
                    # corrupt annotation; ignore
                    pass

            # find intersection with project images
            imgs_match = self.match_filenames(images)

            # filter valid images: found in database and with valid annotation
            imgs_valid = []
            for idx, img in enumerate(imgs_match):
                if img is None:
                    warnings.append(
                        f'"{images[idx]}" could not be found in project.'
                    )
                    continue
                
                # get corresponding annotations
                anno = annotations.get(images[idx], None)
                if anno is None or not len(anno):
                    warnings.append(
                        f'"{images[idx]}": no annotations registered.'
                    )
                    continue
                
                if skipUnknownClasses:
                    # check for validity of label classes
                    anno = [a for a in anno if a['category_id'] in labelClasses]
                    if not len(anno):
                        warnings.append(
                            f'"{images[idx]}": annotations contain label classes that are not registered in project.'
                        )
                        continue
                
                # retrieve image dimensions
                try:
                    imageSize = drivers.load_from_disk(img[1]).shape       # img[1] is the file path as registered in the database
                    print('TODO: debug: check bbox sizes')
                except:
                    # image could not be loaded
                    errors.append(
                        f'"{images[idx]}" could not be loaded.'
                    )
                    continue

                # scale annotations to relative [0,1) coordinates
                for a in range(len(anno)):
                    print('TODO')

                    if clipBoundingBoxes:
                        # limit bounding boxes to image width and height
                        #TODO: we load the images from disk to do so. This is more robust, but might be a bit slow...
                        print('debug')
                    
                # everything alright so far
                imgs_valid.append([img, anno])

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
                # replace label class names with IDs
                for l in range(len(imgs_valid)):
                    for a in range(len(imgs_valid[l][1])):  # annotations
                        catID = imgs_valid[l][a][1]['category_id']
                        lcName = labelClasses[catID][0]
                        imgs_valid[l][a][1]['category_id'] = labelClasses[lcName]
                
                # finally, add annotations to database
