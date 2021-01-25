'''
    Middleware layer for the data administration
    module.
    Responsible for the following tasks and operations:
    - image management: upload and deletion to and from disk
    - annotation and prediction management: up- and download
      of annotations and model predictions

    2020-21 Benjamin Kellenberger
'''

import uuid
import html
from . import celery_interface
from .dataWorker import DataWorker
from modules.Database.app import Database


class DataAdministrationMiddleware:

    def __init__(self, config, dbConnector, taskCoordinator):
        self.config = config
        self.dbConnector = dbConnector  #Database(config)
        self.taskCoordinator = taskCoordinator
        self.dataWorker = DataWorker(config, dbConnector)

        self.jobs = {}      # dict per project of jobs



    def _submit_job(self, project, username, process):
        return self.taskCoordinator.submitJob(project, username, process, 'FileServer')


    
    def pollStatus(self, project, jobID):
        return self.taskCoordinator.pollStatus(project, jobID)



    def getImageFolders(self, project):
        '''
            Returns a dict representing a hierarchical directory
            tree under which the images are stored for a specific
            project.
        '''

        def _integrateBranch(tree, members):
            if not len(members):
                return tree
            elif members[0] not in tree:
                tree[members[0]] = {}
            if len(members)>1:
                tree[members[0]] = _integrateBranch(tree[members[0]], members[1:])
            return tree

        tree = {}
        folderNames = self.dbConnector.execute('''
            SELECT folder FROM "{schema}".fileHierarchy
            ORDER BY folder ASC;
        '''.format(
            schema=html.escape(project)
        ), None, 'all')
        if folderNames is not None and len(folderNames):
            for f in folderNames:
                folder = f['folder']
                if folder is None or not len(folder):
                    continue
                parents = folder.strip('/').split('/')
                tree = _integrateBranch(tree, parents)
        return tree



    def listImages(self, project, username, folder=None, imageAddedRange=None, lastViewedRange=None,
            viewcountRange=None, numAnnoRange=None, numPredRange=None,
            orderBy=None, order='desc', startFrom=None, limit=None):
        '''
            #TODO: update description
            Returns a list of images, with ID, filename,
            date image was added, viewcount, number of annotations,
            number of predictions, and last time viewed, for a given
            project.
            The list can be filtered by all those properties (e.g. 
            date and time image was added, last checked; number of
            annotations, etc.), as well as limited in length (images
            are sorted by date_added).
        '''
        
        # submit job 
        process = celery_interface.listImages.si(project, folder, imageAddedRange,
                                                lastViewedRange, viewcountRange,
                                                numAnnoRange, numPredRange,
                                                orderBy, order, startFrom, limit)
        
        task_id = self._submit_job(project, username, process)
        return task_id
    


    def uploadImages(self, project, images, existingFiles='keepExisting',
        splitImages=False, splitProperties=None):
        '''
            Image upload is handled directly through the
            dataWorker, without a Celery dispatching bridge.
        '''
        return self.dataWorker.uploadImages(project, images, existingFiles,
                                            splitImages, splitProperties)



    def scanForImages(self, project, username):
        '''
            #TODO: update description
            Searches the project image folder on disk for
            files that are valid, but have not (yet) been added
            to the database.
            Returns a list of paths with files.
        '''

        # submit job
        process = celery_interface.scanForImages.si(project)

        task_id = self._submit_job(project, username, process)
        return task_id



    def addExistingImages(self, project, username, imageList=None):
        '''
            #TODO: update description
            Scans the project folder on the file system
            for images that are physically saved, but not
            (yet) added to the database.
            Adds them to the project's database schema.
            If an imageList iterable is provided, only
            the intersection between identified images on
            disk and in the iterable are added.
            If imageList is a string with contents 'all', all
            untracked images on disk will be added.

            Returns a list of image IDs and file names that
            were eventually added to the project database schema.
        '''

        # submit job
        process = celery_interface.addExistingImages.si(project, imageList)

        task_id = self._submit_job(project, username, process)
        return task_id


    
    def removeImages(self, project, username, imageList, forceRemove=False, deleteFromDisk=False):
        '''
            #TODO: update description
            Receives an iterable of image IDs and removes them
            from the project database schema, including associated
            user views, annotations, and predictions made.
            Only removes entries if no user views, annotations, and
            predictions exist, or else if "forceRemove" is True.
            If "deleteFromDisk" is True, the image files are also
            deleted from the project directory on the file system.

            Returns a list of images that were deleted.
        '''

        # submit job
        process = celery_interface.removeImages.si(project,
                                                    imageList,
                                                    forceRemove,
                                                    deleteFromDisk)

        task_id = self._submit_job(project, username, process)
        return task_id



    def prepareDataDownload(self, project, username, dataType='annotation', userList=None, dateRange=None, extraFields=None, segmaskFilenameOptions=None, segmaskEncoding='rgb'):
        '''
            #TODO: update description
            Polls the database for project data according to the
            specified restrictions:
            - dataType: "annotation" or "prediction"
            - userList: for type "annotation": None (all users) or
                        an iterable of user names
            - dateRange: None (all dates) or two values for a mini-
                         mum and maximum timestamp
            - extraFields: additional DB relation columns to query, such as the
                           browser metadata / user agent. To be supplied as a dict.
                           If None, no extra fields will be queried.
            - segmaskFilenameOptions: for segmentation masks only: None (defaults)
                                      or a dict of fields 'baseName' ("id" or "filename"),
                                      'prefix' (str) and 'suffix' (str) to customize the
                                      segmentation masks' file names.
            - segmaskEncoding: for segmentation masks only: set to 'rgb'
                               to encode pixel values with red-green-blue,
                               or to 'indexed' to assign label class index
                               to pixels.
            
            Creates a file in this machine's temporary directory
            and returns the file name to it.
            Note that in some cases (esp. for semantic segmentation),
            the number of queryable entries may be limited due to
            file size and free disk space restrictions. An upper cei-
            ling is specified in the configuration *.ini file ('TODO')
        '''
        # submit job
        process = celery_interface.prepareDataDownload.si(project,
                                                    dataType,
                                                    userList,
                                                    dateRange,
                                                    extraFields,
                                                    segmaskFilenameOptions,
                                                    segmaskEncoding)

        task_id = self._submit_job(project, username, process)
        return task_id