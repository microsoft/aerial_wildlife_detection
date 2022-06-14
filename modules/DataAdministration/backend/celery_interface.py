'''
    Wrapper for the Celery message broker concerning
    the data management module.
    This module may require some longer running tasks,
    such as the preparation of data to download, or
    the scanning of a directory for untracked images.
    These jobs are dispatched as Celery tasks.
    Function "init_celery_dispatcher" is to be initia-
    lized at launch time with a Celery app instance.

    2020-22 Benjamin Kellenberger
'''

import os
from celery import current_app
from .dataWorker import DataWorker
from modules.Database.app import Database
from util.configDef import Config


# initialize dataWorker
modules = os.environ['AIDE_MODULES']
config = Config()
worker = DataWorker(config, Database(config))

@current_app.task()
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task(name='DataAdministration.verify_images', rate_limit=1)
def verifyImages(projects, quickCheck=True):
    return worker.verifyImages(projects, quickCheck)


@current_app.task(name='DataAdministration.list_images')
def listImages(project, folder=None, imageAddedRange=None, lastViewedRange=None,
        viewcountRange=None, numAnnoRange=None, numPredRange=None,
        orderBy=None, order='desc', startFrom=None, limit=None):
    return worker.listImages(project, folder, imageAddedRange, lastViewedRange,
        viewcountRange, numAnnoRange, numPredRange,
        orderBy, order, startFrom, limit)


@current_app.task(name='DataAdministration.scan_for_images')
def scanForImages(project, skipIntegrityCheck=False):
    return worker.scanForImages(project, skipIntegrityCheck=skipIntegrityCheck)


@current_app.task(name='DataAdministration.add_existing_images')
def addExistingImages(project, imageList=None, skipIntegrityCheck=False):
    return worker.addExistingImages(project, imageList, skipIntegrityCheck=skipIntegrityCheck)


@current_app.task(name='DataAdministration.remove_images')
def removeImages(project, imageList, forceRemove=False, deleteFromDisk=False):
    return worker.removeImages(project, imageList, forceRemove, deleteFromDisk)

@current_app.task(name='DataAdministration.request_annotations')
def requestAnnotations(project, username, exportFormat, dataType='annotation', authorList=None, dateRange=None, ignoreImported=False, parserArgs={}):
    return worker.requestAnnotations(project, username, exportFormat, dataType, authorList, dateRange, ignoreImported, parserArgs)

# deprecated
@current_app.task(name='DataAdministration.prepare_data_download')
def prepareDataDownload(project, dataType='annotation', userList=None, dateRange=None, extraFields=None, segmaskFilenameOptions=None, segmaskEncoding='rgb'):
    return worker.prepareDataDownload(project, dataType, userList, dateRange, extraFields, segmaskFilenameOptions, segmaskEncoding)


@current_app.task(name='DataAdministration.watch_image_folders', rate_limit=1)
def watchImageFolders():
    return worker.watchImageFolders()


@current_app.task(name='DataAdministration.delete_project')
def deleteProject(project, deleteFiles=False):
    return worker.deleteProject(project, deleteFiles)