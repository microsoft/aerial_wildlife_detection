'''
    Middleware layer for the data administration
    module.
    Responsible for the following tasks and operations:
    - image management: upload and deletion to and from disk
    - annotation and prediction management: up- and download
      of annotations and model predictions

    2020 Benjamin Kellenberger
'''

import os
import glob
from psycopg2 import sql
from modules.Database.app import Database
from util.helpers import valid_image_extensions


class DataAdministrationMiddleware:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
    

    @staticmethod
    def _scan_dir_imgs(fileDir):
        imgs_disk = set()
        if not fileDir.endswith(os.sep):
            fileDir += os.sep
        files_disk = glob.glob(os.path.join(fileDir, '**'), recursive=True)       #TODO: avoid circular links
        for f in files_disk:
            if os.path.isdir(f):
                continue
            _, ext = os.path.splitext(f)
            if ext.lower() not in valid_image_extensions:
                continue
            fname = f.replace(fileDir, '')
            imgs_disk.add(fname)
        return imgs_disk



    ''' Image administration functionalities '''
    def listImages(self, project, imageAddedRange=None, lastViewedRange=None,
            viewcountRange=None, numAnnoRange=None, numPredRange=None, limit=None):
        '''
            Returns a list of images, with ID, filename,
            date image was added, viewcount, number of annotations,
            number of predictions, and last time viewed, for a given
            project.
            The list can be filtered by all those properties (e.g. 
            date and time image was added, last checked; number of
            annotations, etc.), as well as limited in length (images
            are sorted by date_added).
        '''
        queryArgs = []

        filterStr = ''
        if imageAddedRange is not None:     #TODO
            filterStr += ' date_added >= to_timestamp(%s) AND date_added <= to_timestamp(%s) '
            queryArgs.append(imageAddedRange[0])
            queryArgs.append(imageAddedRange[1])
        if lastViewedRange is not None:     #TODO
            filterStr += 'AND last_viewed >= to_timestamp(%s) AND last_viewed <= to_timestamp(%s) '
            queryArgs.append(lastViewedRange[0])
            queryArgs.append(lastViewedRange[1])
        if viewcountRange is not None:
            filterStr += 'AND viewcount >= %s AND viewcount <= %s '
            queryArgs.append(viewcountRange[0])
            queryArgs.append(viewcountRange[1])
        if numAnnoRange is not None:
            filterStr += 'AND num_anno >= %s AND numAnno <= %s '
            queryArgs.append(numAnnoRange[0])
            queryArgs.append(numAnnoRange[1])
        if numPredRange is not None:
            filterStr += 'AND num_pred >= %s AND num_pred <= %s '
            queryArgs.append(numPredRange[0])
            queryArgs.append(numPredRange[1])

        if filterStr.startswith('AND'):
            filterStr = filterStr[3:]
        if len(filterStr.strip()):
            filterStr = 'WHERE ' + filterStr
        filterStr = sql.SQL(filterStr)

        limitStr = sql.SQL('')
        if isinstance(limit, int):
            limitStr = sql.SQL('LIMIT %s')
            queryArgs.append(limit)

        if not len(queryArgs):
            queryArgs = None

        queryStr = sql.SQL('''
            SELECT img.id, filename, EXTRACT(epoch FROM date_added) AS date_added,
                COALESCE(viewcount, 0) AS viewcount,
                EXTRACT(epoch FROM last_viewed) AS last_viewed,
                COALESCE(num_anno, 0) AS num_anno,
                COALESCE(num_pred, 0) AS num_pred
            FROM {id_img} AS img
            FULL OUTER JOIN (
                SELECT image, COUNT(*) AS viewcount, MAX(last_checked) AS last_viewed
                FROM {id_iu}
                GROUP BY image
            ) AS iu
            ON img.id = iu.image
            FULL OUTER JOIN (
                SELECT image, COUNT(*) AS num_anno
                FROM {id_anno}
                GROUP BY image
            ) AS anno
            ON img.id = anno.image
            FULL OUTER JOIN (
                SELECT image, COUNT(*) AS num_pred
                FROM {id_pred}
                GROUP BY image
            ) AS pred
            ON img.id = pred.image
            ORDER BY date_added DESC
            {filter}
            {limit}
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user'),
            id_anno=sql.Identifier(project, 'annotation'),
            id_pred=sql.Identifier(project, 'prediction'),
            filter=filterStr,
            limit=limitStr
        )
        result = self.dbConnector.execute(queryStr, tuple(queryArgs), 'all')
        for idx in range(len(result)):
            result[idx]['id'] = str(result[idx]['id'])
        return result


    def uploadImages(self, project, images):
        #TODO
        raise Exception('Not yet implemented.')


    def scanForImages(self, project):
        '''
            Searches the project image folder on disk for
            files that are valid, but have not (yet) been added
            to the database.
            Returns a list of paths with files.
        '''
        # scan disk for files
        projectFolder = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), project)
        imgs_disk = self._scan_dir_imgs(projectFolder)
        
        # get all existing file paths from database
        imgs_database = set()
        queryStr = sql.SQL('''
            SELECT filename FROM {id_img};
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        result = self.dbConnector.execute(queryStr, None, 'all')
        for r in range(len(result)):
            imgs_database.add(result[r]['filename'])

        # filter
        imgs_candidates = imgs_disk.difference(imgs_database)
        return imgs_candidates


    def addExistingImages(self, project, imageList=None):
        '''
            Scans the project folder on the file system
            for images that are physically saved, but not
            (yet) added to the database.
            Adds them to the project's database schema.
            If an imageList iterable is provided, only
            the intersection between identified images on
            disk and in the iterable are added.

            Returns a list of image IDs and file names that
            were eventually added to the project database schema.
        '''
        # get all images on disk that are not in database
        imgs_candidates = self.scanForImages(project)

        if imageList is None:
            imgs_add = imgs_candidates
        else:
            imgs_add = imgs_candidates.intersection(set(imageList))
        imgs_add = list(imgs_add)

        # add to database
        queryStr = sql.SQL('''
            INSERT INTO {id_img} (filename)
            VALUES %s;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        self.dbConnector.insert(queryStr, (imgs_add,))

        # get IDs of newly added images
        queryStr = sql.SQL('''
            SELECT id, filename FROM {id_img}
            WHERE filename IN %s;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        result = self.dbConnector.execute(queryStr, (imgs_add,), 'all')

        return result


    def removeImages(self, project, imageList, forceRemove=False, deleteFromDisk=False):
        '''
            Receives an iterable of image IDs and removes them
            from the project database schema, including associated
            user views, annotations, and predictions made.
            Only removes entries if no user views, annotations, and
            predictions exist, or else if "forceRemove" is True.
            If "deleteFromDisk" is True, the image files are also
            deleted from the project directory on the file system.

            Returns a list of image IDs and file names that were
            eventually removed.
        '''


        raise Exception('Not yet implemented.')