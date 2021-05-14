'''
    Wrapper for the Celery message broker concerning
    the Model Marketplace module.

    2020-21 Benjamin Kellenberger
'''

import os
from celery import current_app
from .marketplaceWorker import ModelMarketplaceWorker
from modules.Database.app import Database
from util.configDef import Config


# initialize Model Marketplace worker
modules = os.environ['AIDE_MODULES']
config = Config()
worker = ModelMarketplaceWorker(config, Database(config))


@current_app.task(name='ModelMarketplace.shareModel')
def share_model(project, username, modelID, modelName,
                    modelDescription, tags,
                    public, anonymous):
    return worker.shareModel(project, username, modelID, modelName,
                    modelDescription, tags,
                    public, anonymous)


@current_app.task(name='ModelMarketplace.importModelDatabase')
def import_model_database(modelID, project, username):
    return worker.importModelDatabase(project, username, modelID)


@current_app.task(name='ModelMarketplace.importModelURI')
def import_model_uri(project, username, modelURI, public=True, anonymous=False, forceReimport=True, namePolicy='skip', customName=None):
    return worker.importModelURI(project, username, modelURI, public, anonymous, forceReimport, namePolicy, customName)


@current_app.task(name='ModelMarketplace.requestModelDownload')
def request_model_download(project, username, modelID, source='marketplace', modelName=None, modelDescription='', modelTags=[]):
    return worker.prepareModelDownload(project, modelID, username, source, modelName, modelDescription, modelTags)