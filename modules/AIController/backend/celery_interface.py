'''
    Wrapper for the Celery message broker concerning
    the AIController.

    2020 Benjamin Kellenberger
'''

from celery import current_app
from modules.AIController.backend.middleware import AIMiddleware
from util.configDef import Config


# init AIController middleware
aim = AIMiddleware(Config())


# @current_app.task()
# def aide_internal_notify(message):
#     return aim.aide_internal_notify(message)


@current_app.task()
def start_training(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
    return aim.start_training(project, minTimestamp, minNumAnnoPerImage, maxNumImages, maxNumWorkers)


@current_app.task()
def start_inference(self, project, forceUnlabeled=True, maxNumImages=-1, maxNumWorkers=-1):
    return aim.start_inference(project, forceUnlabeled, maxNumImages, maxNumWorkers)


@current_app.task()
def inference_fixed(self, project, imageIDs, maxNumWorkers=-1):
    return aim.inference_fixed(project, imageIDs, maxNumWorkers)


@current_app.task()
def start_train_and_inference(self, project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=None, 
                                    maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=None, maxNumWorkers_inference=1):
    return aim.start_train_and_inference(project, minTimestamp, minNumAnnoPerImage, maxNumImages_train, 
                                    maxNumWorkers_train,
                                    forceUnlabeled_inference, maxNumImages_inference, maxNumWorkers_inference)