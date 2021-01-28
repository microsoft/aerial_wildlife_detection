'''
    Wrapper for the Celery message broker concerning
    the AIController.

    2020-21 Benjamin Kellenberger
'''

import os
from celery import current_app

from modules.AIController.backend.functional import AIControllerWorker
from util.configDef import Config


# init AIController middleware
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('aicontroller' in modules.lower())
aim = AIControllerWorker(Config(), current_app)



@current_app.task(name='AIController.get_training_images')
def get_training_images(blank, project, epoch, numEpochs, minTimestamp='lastState', includeGoldenQuestions=True, minNumAnnoPerImage=0, maxNumImages=None, numWorkers=1):
    return aim.get_training_images(project, epoch, numEpochs, minTimestamp, includeGoldenQuestions, minNumAnnoPerImage, maxNumImages, numWorkers)


@current_app.task(name='AIController.get_inference_images')
def get_inference_images(blank, project, epoch, numEpochs, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None, numWorkers=1):
    return aim.get_inference_images(project, epoch, numEpochs, goldenQuestionsOnly, forceUnlabeled, maxNumImages, numWorkers)


@current_app.task(name='AIController.delete_model_states')
def delete_model_states(project, modelStateIDs):
    return aim.delete_model_states(project, modelStateIDs)