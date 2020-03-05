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

from celery import current_app
from .dataWorker import DataWorker
from util.configDef import Config


# initialise dataWorker
worker = DataWorker(Config())

@current_app.task()
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task()
def listImages(project, imageAddedRange=None, lastViewedRange=None,
        viewcountRange=None, numAnnoRange=None, numPredRange=None,
        orderBy=None, order='desc', limit=None):
    return worker.listImages(project, imageAddedRange, lastViewedRange,
        viewcountRange, numAnnoRange, numPredRange,
        orderBy, order, limit)


@current_app.task()
def uploadImages(project, images):
    #TODO: check if makes sense to do this in a Celery task
    return worker.uploadImages(project, images)


@current_app.task()
def scanForImages(project):
    return worker.scanForImages(project)


@current_app.task()
def addExistingImages(project, imageList=None):
    return worker.addExistingImages(project, imageList)


@current_app.task()
def removeImages(self, project, imageList, forceRemove=False, deleteFromDisk=False):
    return worker.removeImages(project, imageList, forceRemove, deleteFromDisk)


@current_app.task()
def prepareDataDownload(self, project, dataType='annotation', userList=None, dateRange=None):
    return worker.prepareDataDownload(project, dataType, userList, dateRange)