'''
    Class handling long-running data management-
    related tasks, such as preparing annotations
    for downloading, or scanning directories for
    untracked images.

    2020 Benjamin Kellenberger
'''

import os
import io
import re
import glob
import tempfile
import zipfile
import zlib
from datetime import datetime
import pytz
from uuid import UUID
from celery import current_app
from kombu import Queue
from PIL import Image
from psycopg2 import sql
from modules.Database.app import Database
from modules.LabelUI.backend.annotation_sql_tokens import QueryStrings_annotation, QueryStrings_prediction
from util.helpers import valid_image_extensions, listDirectory, base64ToImage


class DataWorker:

    def __init__(self, config):
        self.config = config
        self.dbConnector = Database(config)
        self.countPattern = re.compile('\_[0-9]+$')
        self._init_project_queues()


    def _init_project_queues(self):
        '''
            Queries the database for projects
            and adds respective queues to Celery to listen to them.
            TODO: implement custom routings
        '''
        if current_app.conf.task_queues is not None:
            queues = list(current_app.conf.task_queues)
            current_queue_names = set([c.name for c in queues])
            print('Existing queues: {}'.format(', '.join([c.name for c in queues])))
        else:
            queues = []
            current_queue_names = set()
        
        log_updates = []
        projects = self.dbConnector.execute('SELECT shortname FROM aide_admin.project WHERE demomode = FALSE;', None, 'all')
        if len(projects):
            for project in projects:
                pName = project['shortname'] + '_dataMgmt'
                if not pName in current_queue_names:
                    current_queue_names.add(pName)
                    queues.append(Queue(pName))
                    current_app.control.add_consumer(pName)     #TODO: stalls if Celery app is not set up. Might be ok to just skip this
                    log_updates.append(pName)
            
            current_app.conf.update(task_queues=tuple(queues))
            print('Added queue(s) for project(s): {}'.format(', '.join(log_updates)))



    def aide_internal_notify(self, message):
        '''
            Used for AIde administrative communication,
            e.g. for setting up queues.
        '''
        if 'task' in message:
            if message['task'] == 'add_projects':
                self._init_project_queues()

            elif message['task'] == 'create_project_folders':
                # set up folders for a newly created project
                if 'projectName' in message:
                    destPath = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), message['projectName'])
                    os.makedirs(destPath, exist_ok=True)

                    # also add project queue
                    self._init_project_queues()



    ''' Image administration functionalities '''
    def listImages(self, project, imageAddedRange=None, lastViewedRange=None,
            viewcountRange=None, numAnnoRange=None, numPredRange=None,
            orderBy=None, order='desc', limit=None):
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

        orderStr = sql.SQL('')
        if orderBy is not None:
            orderStr = sql.SQL('ORDER BY {} {}').format(
                sql.SQL(orderBy),
                sql.SQL(order)
            )

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
                COALESCE(num_pred, 0) AS num_pred,
                img.isGoldenQuestion
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
            {filter}
            {order}
            {limit}
        ''').format(
            id_img=sql.Identifier(project, 'image'),
            id_iu=sql.Identifier(project, 'image_user'),
            id_anno=sql.Identifier(project, 'annotation'),
            id_pred=sql.Identifier(project, 'prediction'),
            filter=filterStr,
            order=orderStr,
            limit=limitStr
        )
        result = self.dbConnector.execute(queryStr, tuple(queryArgs), 'all')
        for idx in range(len(result)):
            result[idx]['id'] = str(result[idx]['id'])
        return result


    def uploadImages(self, project, images):
        '''
            Receives a dict of files (bottle.py file format),
            verifies their file extension and checks if they
            are loadable by PIL.
            If they are, they are saved to disk in the project's
            image folder, and registered in the database.
            Returns image keys for images that were successfully
            saved, and keys and error messages for those that
            were not.
        '''
        imgPaths_valid = []
        imgs_valid = []
        imgs_warn = {}
        imgs_error = {}
        for key in images.keys():
            try:
                nextUpload = images[key]

                #TODO: check if raw_filename is compatible with uploads made from Windows

                # check if correct file suffix
                _, ext = os.path.splitext(nextUpload.raw_filename)
                if not ext.lower() in valid_image_extensions:
                    raise Exception('Invalid file type (*{})'.format(ext))
                
                # check if loadable as image
                cache = io.BytesIO()
                nextUpload.save(cache)
                try:
                    success = Image.open(cache)
                    success.close()
                except Exception:
                    raise Exception('File is not a valid image.')

                parent, filename = os.path.split(nextUpload.raw_filename)
                destFolder = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), project, parent)
                os.makedirs(destFolder, exist_ok=True)
                absFilePath = os.path.join(destFolder, filename)

                # check if an image with the same name does not already exist
                newFileName = filename
                while(os.path.exists(absFilePath)):
                    # rename file
                    fn, ext = os.path.splitext(newFileName)
                    match = self.countPattern.search(fn)
                    if match is None:
                        newFileName = fn + '_1' + ext
                    else:
                        # parse number
                        number = int(fn[match.span()[0]+1:match.span()[1]])
                        newFileName = fn[:match.span()[0]] + '_' + str(number+1) + ext

                    absFilePath = os.path.join(destFolder, newFileName)
                    if not os.path.exists(absFilePath):
                        imgs_warn[key] = 'An image with name "{}" already exists under given path on disk. Image has been renamed to "{}".'.format(
                            filename, newFileName
                        )
                
                # write to disk
                fileParent, _ = os.path.split(absFilePath)
                if len(fileParent):
                    os.makedirs(fileParent, exist_ok=True)
                nextUpload.save(absFilePath)

                imgs_valid.append(key)
                imgPaths_valid.append(os.path.join(parent, newFileName))

            except Exception as e:
                imgs_error[key] = str(e)

        # register valid images in database
        if len(imgPaths_valid):
            queryStr = sql.SQL('''
                INSERT INTO {id_img} (filename)
                VALUES %s;
            ''').format(
                id_img=sql.Identifier(project, 'image')
            )
            self.dbConnector.insert(queryStr, [(i,) for i in imgPaths_valid])

        result = {
            'imgs_valid': imgs_valid,
            'imgPaths_valid': imgPaths_valid,
            'imgs_warn': imgs_warn,
            'imgs_error': imgs_error
        }

        return result


    def scanForImages(self, project):
        '''
            Searches the project image folder on disk for
            files that are valid, but have not (yet) been added
            to the database.
            Returns a list of paths with files.
        '''
        
        # scan disk for files
        projectFolder = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), project)
        if not os.path.isdir(projectFolder) or not os.path.islink(projectFolder):
            # no folder exists for the project (should not happen due to broadcast at project creation)
            return []
        imgs_disk = listDirectory(projectFolder, recursive=True)
        
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
        return list(imgs_candidates)


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
            imgs_add = list(set(imgs_candidates).intersection(set(imageList)))

        # add to database
        queryStr = sql.SQL('''
            INSERT INTO {id_img} (filename)
            VALUES %s;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        self.dbConnector.insert(queryStr, [(i,) for i in imgs_add])     #TODO: incorrect

        # get IDs of newly added images
        queryStr = sql.SQL('''
            SELECT id, filename FROM {id_img}
            WHERE filename IN %s;
        ''').format(
            id_img=sql.Identifier(project, 'image')
        )
        result = self.dbConnector.execute(queryStr, (imgs_add,), 'all')

        status = (0 if len(result) else 1)  #TODO
        return status, result


    def removeImages(self, project, imageList, forceRemove=False, deleteFromDisk=False):
        '''
            Receives an iterable of image IDs and removes them
            from the project database schema, including associated
            user views, annotations, and predictions made.
            Only removes entries if no user views, annotations, and
            predictions exist, or else if "forceRemove" is True.
            If "deleteFromDisk" is True, the image files are also
            deleted from the project directory on the file system.

            Returns a list of images that were deleted.
        '''
        
        imageList = tuple([(UUID(i),) for i in imageList])

        queryArgs = []
        deleteArgs = []
        if forceRemove:
            queryStr = sql.SQL('''
                SELECT id, filename
                FROM {id_img}
                WHERE id IN %s;
            ''').format(
                id_img=sql.Identifier(project, 'image')
            )
            queryArgs = tuple([imageList])

            deleteStr = sql.SQL('''
                DELETE FROM {id_iu} WHERE image IN %s;
                DELETE FROM {id_anno} WHERE image IN %s;
                DELETE FROM {id_pred} WHERE image IN %s;
                DELETE FROM {id_img} WHERE id IN %s;
            ''').format(
                id_iu=sql.Identifier(project, 'image_user'),
                id_anno=sql.Identifier(project, 'annotation'),
                id_pred=sql.Identifier(project, 'prediction'),
                id_img=sql.Identifier(project, 'image')
            )
            deleteArgs = tuple([imageList] * 4)
        
        else:
            queryStr = sql.SQL('''
                SELECT id, filename
                FROM {id_img}
                WHERE id IN %s
                AND id NOT IN (
                    SELECT image FROM {id_iu}
                    WHERE image IN %s
                    UNION ALL
                    SELECT image FROM {id_anno}
                    WHERE image IN %s
                    UNION ALL
                    SELECT image FROM {id_pred}
                    WHERE image IN %s
                );
            ''').format(
                id_img=sql.Identifier(project, 'image'),
                id_iu=sql.Identifier(project, 'image_user'),
                id_anno=sql.Identifier(project, 'annotation'),
                id_pred=sql.Identifier(project, 'prediction')
            )
            queryArgs = tuple([imageList] * 4)

            deleteStr = sql.SQL('''
                DELETE FROM {id_img}
                WHERE id IN %s
                AND id NOT IN (
                    SELECT image FROM {id_iu}
                    WHERE image IN %s
                    UNION ALL
                    SELECT image FROM {id_anno}
                    WHERE image IN %s
                    UNION ALL
                    SELECT image FROM {id_pred}
                    WHERE image IN %s
                );
            ''').format(
                id_img=sql.Identifier(project, 'image'),
                id_iu=sql.Identifier(project, 'image_user'),
                id_anno=sql.Identifier(project, 'annotation'),
                id_pred=sql.Identifier(project, 'prediction')
            )
            deleteArgs = tuple([imageList] * 4)

        # retrieve images to be deleted
        imgs_del = self.dbConnector.execute(queryStr, queryArgs, 'all')

        if imgs_del is None:
            imgs_del = []

        if len(imgs_del):
            # delete images
            self.dbConnector.execute(deleteStr, deleteArgs, None)

            if deleteFromDisk:
                projectFolder = os.path.join(self.config.getProperty('FileServer', 'staticfiles_dir'), project)
                if os.path.isdir(projectFolder) or os.path.islink(projectFolder):
                    for i in imgs_del:
                        filePath = os.path.join(projectFolder, i['filename'])
                        if os.path.isfile(filePath):
                            os.remove(filePath)

            # convert UUID
            for idx in range(len(imgs_del)):
                imgs_del[idx]['id'] = str(imgs_del[idx]['id'])

        return imgs_del


    def prepareDataDownload(self, project, dataType='annotation', userList=None, dateRange=None):
        '''
            Polls the database for project data according to the
            specified restrictions:
            - dataType: "annotation" or "prediction"
            - userList: for type "annotation": None (all users) or
                        an iterable of user names
            - dateRange: None (all dates) or two values for a mini-
                         mum and maximum timestamp
            
            Creates a file in this machine's temporary directory
            and returns the file name to it.
            Note that in some cases (esp. for semantic segmentation),
            the number of queryable entries may be limited due to
            file size and free disk space restrictions. An upper cei-
            ling is specified in the configuration *.ini file ('TODO')
        '''

        now = datetime.now(tz=pytz.utc)

        # argument check
        if userList is None:
            userList = []
        elif isinstance(userList, str):
            userList = [userList]
        if dateRange is None:
            dateRange = []
        elif len(dateRange) == 1:
            dateRange = [dateRange, now]

        # check metadata type: need to deal with segmentation masks separately
        if dataType == 'annotation':
            metaField = 'annotationtype'
        elif dataType == 'prediction':
            metaField = 'predictiontype'
        else:
            raise Exception('Invalid dataType specified ({})'.format(dataType))
        metaType = self.dbConnector.execute('''
                SELECT {} FROM aide_admin.project
                WHERE shortname = %s;
            '''.format(metaField),
            (project,),
            1
        )[0][metaField]

        if metaType.lower() == 'segmentationmasks':
            is_segmentation = True
            fileExtension = '.zip'
        else:
            is_segmentation = False
            fileExtension = '.txt'      #TODO: support JSON?

        # prepare output file
        filename = 'aide_query_{}'.format(now.strftime('%Y-%m-%d_%H-%M-%S')) + fileExtension
        destPath = os.path.join(tempfile.gettempdir(), 'aide/downloadRequests', project)
        os.makedirs(destPath, exist_ok=True)
        destPath = os.path.join(destPath, filename)

        # generate query
        queryArgs = []
        tableID = sql.Identifier(project, dataType)
        userStr = sql.SQL('')
        iuStr = sql.SQL('')
        dateStr = sql.SQL('')
        queryFields = []
        if dataType == 'annotation':
            iuStr = sql.SQL('''
                JOIN (SELECT image AS iu_image, username AS iu_username, viewcount, last_checked, last_time_required FROM {id_iu}) AS iu
                ON t.image = iu.iu_image
                AND t.username = iu.iu_username
            ''').format(
                id_iu=sql.Identifier(project, 'image_user')
            )
            if len(userList):
                userStr = sql.SQL('WHERE username IN %s')
                queryArgs.append(tuple(userList))
            
            queryFields.extend(getattr(QueryStrings_annotation, metaType).value)
            queryFields.extend(['username', 'viewcount', 'last_checked', 'last_time_required']) #TODO: make customizable

        else:
            queryFields.extend(getattr(QueryStrings_prediction, metaType).value)

        if len(dateRange):
            if len(userStr.string):
                dateStr = sql.SQL(' AND timecreated >= to_timestamp(%s) AND timecreated <= to_timestamp(%s)')
            else:
                dateStr = sql.SQL('WHERE timecreated >= to_timestamp(%s) AND timecreated <= to_timestamp(%s)')
            queryArgs.extend(dateRange)

        queryStr = sql.SQL('''
            SELECT * FROM {tableID} AS t
            {iuStr}
            {userStr}
            {dateStr}
        ''').format(
            tableID=tableID,
            iuStr=iuStr,
            userStr=userStr,
            dateStr=dateStr
        )

        # query and process data
        if is_segmentation:
            mainFile = zipfile.ZipFile(destPath, 'w', zipfile.ZIP_DEFLATED)
        else:
            mainFile = open(destPath, 'w')
        metaStr = '; '.join(queryFields) + '\n'

        with self.dbConnector.execute_cursor(queryStr, tuple(queryArgs)) as cursor:
            while True:
                b = cursor.fetchone()
                if b is None:
                    break

                if is_segmentation:
                    # convert and store segmentation mask separately
                    segmask_filename = 'segmentation_masks/' + str(b['image']) + '.tif'
                    segmask = base64ToImage(b['segmentationmask'], b['width'], b['height'])
                    bio = io.BytesIO()
                    segmask.save(bio, 'TIFF')       #TODO: file format
                    mainFile.writestr(segmask_filename, segmask.getvalue())

                # store metadata
                metaLine = ''
                for field in queryFields:
                    if field.lower() == 'segmentationmask':
                        continue
                    metaLine += '{}; '.format(b[field.lower()])
                metaStr += metaLine + '\n'
        
        if is_segmentation:
            mainFile.writestr('query.txt', metaStr)
        else:
            mainFile.write(metaStr)

        mainFile.close()

        return filename