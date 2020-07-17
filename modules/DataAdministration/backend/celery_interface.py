'''
    Wrapper for the Celery message broker concerning
    the data management module.
    This module may require some longer running tasks,
    such as the preparation of data to download, or
    the scanning of a directory for untracked images.
    These jobs are dispatched as Celery tasks.
    Function "init_celery_dispatcher" is to be initia-
    lized at launch time with a Celery app instance.

    2020 Benjamin Kellenberger
'''

import os
from celery import current_app
from .dataWorker import DataWorker
from util.configDef import Config


# initialise dataWorker
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('fileserver' in modules.lower())
worker = DataWorker(Config())

@current_app.task()
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task(name='DataAdministration.list_images')
def listImages(project, imageAddedRange=None, lastViewedRange=None,
        viewcountRange=None, numAnnoRange=None, numPredRange=None,
        orderBy=None, order='desc', startFrom=None, limit=None):
    return worker.listImages(project, imageAddedRange, lastViewedRange,
        viewcountRange, numAnnoRange, numPredRange,
        orderBy, order, startFrom, limit)


# @current_app.task(name='DataAdministration.upload_images')
# def uploadImages(project, images):
#     #TODO: check if makes sense to do this in a Celery task
#     return worker.uploadImages(project, images)


@current_app.task(name='DataAdministration.scan_for_images')
def scanForImages(project):
    return worker.scanForImages(project)


@current_app.task(name='DataAdministration.add_existing_images')
def addExistingImages(project, imageList=None):
    return worker.addExistingImages(project, imageList)


@current_app.task(name='DataAdministration.remove_images')
def removeImages(project, imageList, forceRemove=False, deleteFromDisk=False):
    return worker.removeImages(project, imageList, forceRemove, deleteFromDisk)


@current_app.task(name='DataAdministration.prepare_data_download')
def prepareDataDownload(project, dataType='annotation', userList=None, dateRange=None, extraFields=None, segmaskFilenameOptions=None, segmaskEncoding='rgb'):
    return worker.prepareDataDownload(project, dataType, userList, dateRange, extraFields, segmaskFilenameOptions, segmaskEncoding)


@current_app.task(name='DataAdministration.watch_image_folders', rate_limit=1)
def watchImageFolders():
    return worker.watchImageFolders()