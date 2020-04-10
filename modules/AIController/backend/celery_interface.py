'''
    Wrapper for the Celery message broker concerning
    the AIController.

    2020 Benjamin Kellenberger
'''

import os
from celery import current_app
from modules.AIController.backend.middleware import AIMiddleware
from util.configDef import Config


# init AIController middleware
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('aicontroller' in modules.lower())
aim = AIMiddleware(Config(), passiveMode)


#TODO
@current_app.task
def add(x, y):
    print("Adding {} + {}".format(x, y))
    return x + y
@current_app.task(name='Dummy_a')
def dummy_a(num_workers=1):
    print("I would split according to {}, but I just return three splits.".format(num_workers))
    return [[1,2,3],[4,5,6,7],[8,9,10]]
@current_app.task(name='Dummy_b')
def dummy_b(imageIDs, index):
    print("My Index is {}".format(index))
    print("I received the following image IDs:")
    print(imageIDs)
    print("So I return the following subset:")
    print(imageIDs[index])
    if index == 1:
        print("I am a nasty application, I wait 4 extra seconds")
        import time
        time.sleep(4)
    return imageIDs[index],index
@current_app.task(name='Dummy_c')
def dummy_c(inputs):
    print("Dummy C here. I got the following input:")
    print(inputs)
    return inputs


@current_app.task(name='AIController.aide_internal_notify')
def aide_internal_notify(message):
    return aim.aide_internal_notify(message)


@current_app.task(name='AIController.get_training_images')
def get_training_images(project, minTimestamp='lastState', includeGoldenQuestions=True, minNumAnnoPerImage=0, maxNumImages=None):
    return aim.get_training_images(project, minTimestamp, includeGoldenQuestions, minNumAnnoPerImage, maxNumImages)


@current_app.task(name='AIController.get_inference_images')
def get_inference_images(project, goldenQuestionsOnly=False, forceUnlabeled=False, maxNumImages=None):
    return aim.get_inference_images(project, goldenQuestionsOnly, forceUnlabeled, maxNumImages)


@current_app.task(name='AIController.start_training')
def start_training(project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages=None, maxNumWorkers=-1):
    return aim.start_training(project, minTimestamp, minNumAnnoPerImage, maxNumImages, maxNumWorkers)


@current_app.task(name='AIController.start_inference')
def start_inference(project, forceUnlabeled=True, maxNumImages=-1, maxNumWorkers=-1):
    return aim.start_inference(project, forceUnlabeled, maxNumImages, maxNumWorkers)


@current_app.task(name='AIController.inference_fixed')
def inference_fixed(project, imageIDs, maxNumWorkers=-1):
    return aim.inference_fixed(project, imageIDs, maxNumWorkers)


@current_app.task(name='AIController.start_train_and_inference')
def start_train_and_inference(project, minTimestamp='lastState', minNumAnnoPerImage=0, maxNumImages_train=None, 
                                    maxNumWorkers_train=1,
                                    forceUnlabeled_inference=True, maxNumImages_inference=None, maxNumWorkers_inference=1):
    return aim.start_train_and_inference(project, minTimestamp, minNumAnnoPerImage, maxNumImages_train, 
                                    maxNumWorkers_train,
                                    forceUnlabeled_inference, maxNumImages_inference, maxNumWorkers_inference)