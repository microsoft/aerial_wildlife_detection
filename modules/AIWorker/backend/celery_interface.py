'''
    Wrapper for the Celery message broker concerning
    the AIWorker(s).

    2019-20 Benjamin Kellenberger
'''

import os
from celery import current_app
from kombu.common import Broadcast
from modules.AIWorker.app import AIWorker
from util.configDef import Config


# init AIWorker
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('aiworker' in modules.lower())
worker = AIWorker(Config(), passiveMode)


@current_app.task(name='AIWorker.aide_internal_notify')
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task(name='AIWorker.call_train', rate_limit=1)
def call_train(data, index, project):
    is_subset = (len(data) > 1)
    if index <= len(data):
        return worker.call_train(data[index], project, is_subset)
    else:
        # worker not needed
        print("Subset {} requested, but only {} chunks provided. Skipping...".format(
            index, len(data)
        ))
        return 0


@current_app.task(name='AIWorker.call_average_model_states', rate_limit=1)
def call_average_model_states(project, *args):
    return worker.call_average_model_states(project)


@current_app.task(name='AIWorker.call_inference')
def call_inference(project, imageIDs):
    return worker.call_inference(project, imageIDs)