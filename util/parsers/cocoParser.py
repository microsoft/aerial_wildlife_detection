'''
    Label parser for annotations in MS-COCO format:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from datetime import datetime
from collections import defaultdict
import re
import difflib
import numpy as np
from psycopg2 import sql

from constants.version import AIDE_VERSION
from modules.ImageQuerying.backend.regionProcessing import polygon_area     #TODO: make generally available?
from util.parsers.abstractParser import AbstractAnnotationParser
from util import helpers, drivers
drivers.init_drivers()


class COCOparser(AbstractAnnotationParser):

    NAME = 'MS-COCO'
    INFO = '<p>Supports annotations of labels, bounding boxes, and polygons in the <a href="https://cocodataset.org/" target="_blank">MS-COCO</a> format.'
    ANNOTATION_TYPES = ('labels', 'boundingBoxes', 'polygons')

    FILE_SUB_PATTERN = '(\/|\\\\)*.*\/*(images|labels|annotations)(\/|\\\\)*'           # pattern that attempts to identify image file name from label file name

    @classmethod
    def get_html_options(cls, method):
        '''
            The COCOparser has various options to set during im- and export.
        '''
        if method == 'import':
            return '''
            <div>
                <input type="checkbox" id="skip_empty_images" />
                <label for="skip_empty_images">skip images without annotations</label>
                <br />
                <input type="checkbox" id="verify_image_size" />
                <label for="verify_image_size">verify image size</label>
                <p style="margin-left:10px;font-style:italic">
                    If checked, images will be loaded from disk and checked for size
                    instead of relying on the size attributes in the annotation files.
                    This may be more accurate, but significantly slower.
                </p>
                <input type="checkbox" id="mark_iscrowd_as_unsure" />
                <label for="mark_iscrowd_as_unsure">mark 'iscrowd' annotations as unsure</label>
            </div>
            '''
        
        else:
            return '''
            <div>
                <input type="checkbox" id="mark_unsure_as_iscrowd" />
                <label for="mark_unsure_as_iscrowd">mark unsure annotations as 'iscrowd'</label>
            </div>
            '''


    @classmethod
    def _get_coco_files(cls, fileDict, folderPrefix):
        '''
            Iterates through keys of a provided dict of file names and returns
            all those that appear to be valid MS-COCO JSON files.
        '''
        cocoFiles = []
        for file_orig in fileDict.keys():
            if isinstance(folderPrefix, str):
                filePath = os.path.join(folderPrefix, file_orig)
            else:
                filePath = file_orig
            if not os.path.isfile(filePath) and not os.path.islink(filePath):
                continue
            _, ext = os.path.splitext(filePath.lower())
            if not ext in ('.json', '.txt', ''):
                continue
            try:
                meta = json.load(open(filePath, 'r'))

                # valid JSON file; check for required fields
                assert 'images' in meta
                assert isinstance(meta['images'], list) or isinstance(meta['images'], tuple)
                assert 'categories' in meta
                assert isinstance(meta['categories'], list) or isinstance(meta['categories'], tuple)
                assert 'annotations' in meta
                assert isinstance(meta['annotations'], list) or isinstance(meta['annotations'], tuple)

                # basic supercategories are present; we assume this is indeed a valid COCO file
                cocoFiles.append(filePath)

            except:
                # unparseable or erroneous format
                continue
        
        return cocoFiles


    @classmethod
    def is_parseable(cls, fileDict, folderPrefix):
        '''
            File dict is parseable if at least one file is a proper MS-COCO JSON
            file.
        '''
        return len(cls._get_coco_files(fileDict, folderPrefix)) > 0


    def import_annotations(self, fileDict, targetAccount, skipUnknownClasses, markAsGoldenQuestions, **kwargs):

        # args setup
        skipEmptyImages = kwargs.get('skip_empty_images', False)    # if True, images with zero annotations will not be added to the "image_user" relation
        verifyImageSize = kwargs.get('verify_image_size', False)    # if True, image sizes will be retrieved from files, not just from COCO metadata
        unsure = kwargs.get('mark_iscrowd_as_unsure', False)        # if True, annotations with attribute "iscrowd" will be marked as "unsure" in AIDE

        now = helpers.current_time()

        uploadedImages = set([f for f in fileDict.keys() if fileDict[f] != '-1'])

        dbFieldNames = []
        if self.annotationType == 'boundingBoxes':
            dbFieldNames = ['x', 'y', 'width', 'height']
        elif self.annotationType == 'polygons':
            dbFieldNames = ['coordinates']
        elif self.annotationType == 'segmentationMasks':
            dbFieldNames = ['segmentationmask', 'width', 'height']

        importedAnnotations, warnings, errors = defaultdict(list), [], []
        imgs_valid = []

        # get valid COCO files
        cocoFiles = self._get_coco_files(fileDict, self.tempDir)
        if not len(cocoFiles):
            return {
                'result': {},
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
                        assert all([a >= 0.0 for a in anno['bbox']])
                        annotations[anno['image_id']].append(anno)
                except:
                    # corrupt annotation; ignore
                    pass

            # find intersection with project images
            imgCandidates = {}
            for key in list(images.keys()):
                # file key to match the annotation file with the image(s) present in the database
                fKey = key
                if fKey.startswith(os.sep) or fKey.startswith('/'):
                    fKey = fKey[1:]
                fKey = re.sub(self.FILE_SUB_PATTERN, '', fKey, flags=re.IGNORECASE)        # replace "<base folder>/(labels|annotations)/*" with "%/*" for search in database with "LIKE" expression
                fKey = os.path.splitext(fKey)[0]

                if len(uploadedImages):
                    candidates = difflib.get_close_matches(fKey, uploadedImages, n=1)
                    if len(candidates):
                        fKey = fileDict[candidates[0]]      # use new, potentially overwritten file name
                imgCandidates[key] = fKey

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
                pl=sql.SQL(','.join(['%s' for _ in imgCandidates.keys()])),
                id_img=sql.Identifier(self.project, 'image'),
                id_iu=sql.Identifier(self.project, 'image_user')
            ),
            tuple((v,) for v in imgCandidates.values()), 'all')
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
            for key in images.keys():
                imgCOCOmeta = images[key]
                imgCOCOid = imgCOCOmeta['id']

                candidateKey = imgCandidates[key]
                if candidateKey not in img_lut:
                    # image got blacklisted or has not been found in database
                    warnings.append(
                        f'Image with ID {imgCOCOid} ({key}) could not be found or contains previously imported annotations; skipped.'
                    )
                    continue
                
                # get corresponding annotations
                anno = annotations.get(imgCOCOid, None)
                if anno is None or not len(anno):
                    warnings.append(
                        f'Image with ID {imgCOCOid} ({key}): no annotations registered.'
                    )
                    if skipEmptyImages:
                        continue

                if skipUnknownClasses:
                    # check for validity of label classes
                    anno = [a for a in anno if a['category_id'] in labelClasses]
                    if not len(anno):
                        warnings.append(
                            f'"Image with ID {imgCOCOid} ({key}): annotations contain label classes that are not registered in project.'
                        )
                        continue
                
                img_candidates = img_lut[candidateKey]

                # create spatial index of image virtual views
                imgViewIndex = {}
                for ic in img_candidates:
                    x, y = ic['y'], ic['x']
                    if y not in imgViewIndex:
                        imgViewIndex[y] = {}
                    imgViewIndex[y][x] = {'id': ic['id'], 'window': [y, x, ic['height'], ic['width']]}

                annos_valid = []
                for a in range(len(anno)):
                    isUnsure = anno[a].get('iscrowd', False) and unsure
                    if self.annotationType == 'boundingBoxes':
                        bbox = [float(v) for v in anno[a]['bbox']]
                        assert len(bbox) == 4, f'invalid number of bounding box coordinates ({len(bbox)} != 4)'
                    
                    elif self.annotationType == 'polygons':
                        # we parse the segmentation information for that
                        seg = [float(v) for v in anno[a]['segmentation']]
                        assert len(seg) % 2 == 0 and len(seg) >= 6, f'invalid number of coordinates for polygon encountered'
                        
                        # close polygon if needed
                        if seg[0] != seg[-2] or seg[1] != seg[-1]:
                            seg.extend(seg[:2])

                    # get image(s)
                    if self.annotationType == 'labels':
                        # assign to all virtual views
                        for i in range(len(img_candidates)):
                            imgID = img_candidates[i]['id']
                            annos_valid.append([
                                imgID,
                                isUnsure,
                                anno[a]['category_id']
                            ])
                    
                    elif self.annotationType == 'polygons':
                            #TODO
                            print('not yet implemented')
                        
                    elif self.annotationType == 'segmentationMasks':
                        #TODO: implement drawing segmasks from polygons
                        raise NotImplementedError('To be implemented.')

                    else:
                        # retrieve correct image (resp. virtual view) for annotation
                        imgIdx_y = np.array(list(imgViewIndex.keys()))
                        yDiff = (bbox[1]+bbox[3]) - imgIdx_y
                        yDiff[yDiff<0] = 1e9
                        matchY = imgIdx_y[np.argmin(yDiff)].tolist()

                        imgIdx_x = np.array(list(imgViewIndex[matchY].keys()))
                        xDiff = (bbox[0]+bbox[2]) - imgIdx_x
                        xDiff[xDiff<0] = 1e9
                        matchX = imgIdx_x[np.argmin(xDiff)].tolist()
                        imgMeta = imgViewIndex[matchY][matchX]
                        window = imgMeta['window']
                        bbox_target = [
                            (bbox[0] - window[1]) / window[3],
                            (bbox[1] - window[0]) / window[2],
                            bbox[2] / window[3],
                            bbox[3] / window[2]
                        ]
                        bbox_target[0] += bbox_target[2]/2.0
                        bbox_target[1] += bbox_target[3]/2.0

                        if any([b < 0 for b in bbox_target]) or any([b > 1 for b in bbox_target]):
                            continue

                        annos_valid.append([
                            imgMeta['id'],
                            isUnsure,
                            anno[a]['category_id'],
                            *bbox_target
                        ])
               
                if len(annos_valid) or not skipEmptyImages:
                    imgs_valid.extend(annos_valid)

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
                for i in range(len(imgs_valid)):
                    # replace label class names with UUIDs
                    labelName = labelClasses[imgs_valid[i][2]][0]
                    labelUUID = self.labelClasses[labelName]
                    imgs_valid[i][2] = labelUUID
                
                # finally, add annotations to database
                result = self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {id_anno} (username, image, timeCreated, timeRequired, unsure, label, {annoFields})
                    VALUES %s
                    RETURNING image, id;
                ''').format(
                    id_anno=sql.Identifier(self.project, 'annotation'),
                    annoFields=sql.SQL(','.join(dbFieldNames))
                ),
                tuple([(targetAccount, i[0], now, -1, i[1],
                    *i[2:]) for i in imgs_valid]), 'all')
                for row in result:
                    importedAnnotations[row[0]].append(row[1])

                # also set in image_user relation
                imgIDs = set(i[0] for i in imgs_valid)
                self.dbConnector.insert(sql.SQL('''
                    INSERT INTO {} (username, image, last_checked, last_time_required)
                    VALUES %s
                    ON CONFLICT (username, image) DO UPDATE
                    SET last_time_required = -1;
                ''').format(sql.Identifier(self.project, 'image_user')),
                tuple([(targetAccount, i, now, -1) for i in imgIDs]))
            
            if markAsGoldenQuestions and len(imgIDs):
                self.dbConnector.insert(sql.SQL('''
                    UPDATE {}
                    SET isGoldenQuestion = TRUE
                    WHERE id IN (%s);
                ''').format(sql.Identifier(self.project, 'image')),
                tuple([(i,) for i in list(imgIDs)]))

        return {
            'result': importedAnnotations,
            'warnings': warnings,
            'errors': errors
        }


    def export_annotations(self, annotations, destination, **kwargs):

        # args
        markUnsureAsIscrowd = kwargs.get('mark_unsure_as_iscrowd', False)       # if True, annotations marked as "unsure" in database will be marked as "iscrowd" in COCO output

        # prepare COCO-formatted output
        try:
            url = self.config.getProperty('Server', 'host') + ':' + self.config.getProperty('Server', 'port') + f'/{self.project}'
        except:
            url = '(unknown)'
        out = {
            'info': {
                'year': datetime.now().year,
                'version': '1.0',
                'description': f'AIDE version {AIDE_VERSION} export for project "{self.project}"',
                'contributor': self.user,
                'url': url,
                'date_created': str(helpers.current_time())
            },
            'licenses': []      #TODO
        }
        categories = []
        images = []
        annotations_out = []

        # get labelclass groups
        lcGroups = self.dbConnector.execute(sql.SQL('''
            SELECT * FROM {};
        ''').format(sql.Identifier(self.project, 'labelclassgroup')), None, 'all')
        lcGroups = dict([[l['id'], l['name']] for l in lcGroups])

        # create new labelclass index for COCO format       #TODO: sort after labelclass name first?
        categoryLookup = {}
        for lc in annotations['labelclasses']:
            catID = len(categories) + 1
            categoryLookup[lc['id']] = catID
            catInfo = {
                'id': catID,
                'name': lc['name']
            }
            if lc['labelclassgroup'] is not None and lc['labelclassgroup'] in lcGroups:
                catInfo['supercategory'] = lcGroups[lc['labelclassgroup']]
            categories.append(catInfo)
        
        # export images
        imgFilenameLookup = {}
        imgIDLookup = {}
        imgSizeLookup = {}
        for img in annotations['images']:
            if img['filename'] in imgFilenameLookup:
                meta = imgFilenameLookup[img['filename']]
                imgIDLookup[img['id']] = meta['id']
                isz = meta['size']
                imgSizeLookup[img['id']] = [img.get('y', None), img.get('x', None), img.get('height', isz[0]), img.get('width', isz[1])]
                continue

            try:
                fname = os.path.join(self.projectRoot, img['filename'])
                driver = drivers.get_driver(fname)
                imsize = driver.size(fname)[1:]
            except:
                continue        #TODO
            
            imgID = len(images) + 1
            imgFilenameLookup[img['filename']] = {'id': imgID, 'size': imsize}
            imgIDLookup[img['id']] = imgID
            patchSize = [img.get('height', imsize[0]), img.get('width', imsize[1])]
            if patchSize[0] is None:
                patchSize[0] = imsize[1]
            if patchSize[1] is None:
                patchSize[1] = imsize[0]
            imgSizeLookup[img['id']] = [img.get('y', None), img.get('x', None), *patchSize]
            images.append({
                'id': imgID,
                'width': imsize[1],
                'height': imsize[0],
                'file_name': img['filename'],
                #TODO: license & Co.: https://cocodataset.org/#format-data
            })
        
        # export annotations
        for anno in annotations['annotations']:
            annoID = len(annotations) + 1
            annoInfo = {
                'id': annoID,
                'image_id': imgIDLookup[anno['image']],
                'category_id': categoryLookup[anno['label']],
                'iscrowd': anno.get('unsure', False) if markUnsureAsIscrowd else False
            }
            if self.annotationType == 'boundingBoxes':
                # convert bounding box back to absolute XYWH format
                imsize = imgSizeLookup[anno['image']]
                bbox = [
                    anno['x']*imsize[2],
                    anno['y']*imsize[3],
                    anno['width']*imsize[2],
                    anno['height']*imsize[3]
                ]
                bbox[0] -= bbox[2]/2.0
                bbox[1] -= bbox[3]/2.0

                if imsize[0] is not None and imsize[1] is not None:
                    # virtual view; shift bbox
                    bbox[0] += imsize[1]
                    bbox[1] += imsize[0]

                annoInfo['bbox'] = bbox
                annoInfo['area'] = bbox[2]*bbox[3]
            
            elif self.annotationType == 'polygons':
                # convert coordinates to absolute format
                imsize = imgSizeLookup[anno['image']]
                isView = imsize[0] is not None and imsize[1] is not None
                coords = anno['coordinates']
                for c in range(len(coords)):
                    coords[c] *= imsize[2+(c+1)%2]
                    if isView:
                        # virtual view; shift coordinates
                        coords[c] += imsize[(c+1)%2]

                annoInfo['segmentation'] = coords
                annoInfo['area'] = polygon_area(coords[1::2], coords[::2]).tolist()
            
            elif self.annotationType == 'segmentationMasks':
                #TODO
                raise NotImplementedError('To be implemented.')

            annotations_out.append(annoInfo)

        # assemble everything into JSON file
        out['categories'] = categories
        out['images'] = images
        out['annotations'] = annotations_out

        coco_dump = json.dumps(out, ensure_ascii=False, indent=4)
        destination.writestr('annotations.json', data=coco_dump)

        return {
            'files': ('annotations.json',)
        }



#TODO
if __name__ == '__main__':

    project = 'cocotest'
    fileDir = '/data/datasets/Kuzikus/patches_800x600'
    targetAccount = 'bkellenb'
    annotationType = 'boundingBoxes'

    from tqdm import tqdm
    import glob
    fileList = glob.glob(os.path.join(fileDir, '**/*'), recursive=True)


    from util.configDef import Config
    from modules.Database.app import Database

    config = Config()
    dbConnector = Database(config)


    if True:
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
        parser = COCOparser(config, dbConnector, project, targetAccount, annotationType)

        kwargs = {
            'skip_empty_images': True
        }

        result = parser.import_annotations(fileList, targetAccount, skipUnknownClasses=False, markAsGoldenQuestions=True, **kwargs)
    

    else:
        # annotation export
        from modules.DataAdministration.backend.dataWorker import DataWorker
        dw = DataWorker(config, dbConnector)
        outFile = dw.requestAnnotations(project, targetAccount, 'mscoco', 'annotation')

        print(outFile)