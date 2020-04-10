'''
    Wrapper for the Celery message broker concerning
    the AIController.

    2020 Benjamin Kellenberger
'''

import os
from celery import current_app
# from modules.AIController.backend.middleware import AIMiddleware
from modules.AIController.backend.functional import AIControllerWorker
from util.configDef import Config


# init AIController middleware
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('aicontroller' in modules.lower())
aim = AIControllerWorker(Config(), current_app)
# aim = AIMiddleware(Config(), passiveMode)




# @current_app.task(name='AIController.aide_internal_notify')
# def aide_internal_notify(message):
#     return aim.aide_internal_notify(message)


@current_app.task(name='AIController.get_training_images')
def get_training_images(project, minTimestamp='lastState', includeGoldenQuestions=True, minNumAnnoPerImage=0, maxNumImages=None, numWorkers=1):
    return aim.get_training_images(project, minTimestamp, includeGoldenQuestions, minNumAnnoPerImage, maxNumImages, numWorkers)


@current_app.task(name='AIController.get_inference_images')
def get_inference_images(project, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, numWorkers=1):
    return aim.get_inference_images(project, goldenQuestionsOnly, forceUnlabeled, maxNumImages, numWorkers)


# @current_app.task(name='AIController.start_training')
# def start_training(project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
#     return aim.start_training(project, minTimestamp, minNumAnnoPerImage, maxNumImages, maxNumWorkers)


# @current_app.task(name='AIController.start_inference')
# def start_inference(project, forceUnlabeled=True, maxNumImages=-1, maxNumWorkers=-1):
#     return aim.start_inference(project, forceUnlabeled, maxNumImages, maxNumWorkers)


# @current_app.task(name='AIController.inference_fixed')
# def inference_fixed(project, imageIDs, maxNumWorkers=-1):
#     return aim.inference_fixed(project, imageIDs, maxNumWorkers)


# @current_app.task(name='AIController.start_train_and_inference')
# def start_train_and_inference(project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=None, 
#                                     maxNumWorkers_train=1,
#                                     forceUnlabeled_inference=True, maxNumImages_inference=None, maxNumWorkers_inference=1):
#     return aim.start_train_and_inference(project, minTimestamp, minNumAnnoPerImage, maxNumImages_train, 
#                                     maxNumWorkers_train,
#                                     forceUnlabeled_inference, maxNumImages_inference, maxNumWorkers_inference)