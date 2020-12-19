'''
    Wrapper for the Celery message broker concerning
    the AIWorker(s).

    2019-20 Benjamin Kellenberger
'''

import os
from celery import current_app
from kombu.common import Broadcast
from constants.version import AIDE_VERSION
from modules.AIWorker.app import AIWorker
from util.configDef import Config


# init AIWorker
modules = os.environ['AIDE_MODULES']
passiveMode = (os.environ['PASSIVE_MODE']=='1' if 'PASSIVE_MODE' in os.environ else False) or not('aiworker' in modules.lower())
worker = AIWorker(Config(), passiveMode)



@current_app.task(name='AIWorker.aide_internal_notify')
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task(name='AIWorker.call_update_model', rate_limit=1)
def call_update_model(blank, numEpochs, project):
    return worker.call_update_model(numEpochs, project)


@current_app.task(name='AIWorker.call_train', rate_limit=1)
def call_train(data, index, epoch, numEpochs, project, aiModelSettings):
    if len(data) == 2 and data[1] is None:
        # model update call preceded training task; ignore empty output of it
        data = data[0]

    is_subset = (len(data) > 1)
    if index < len(data):
        return worker.call_train(data[index], epoch, numEpochs, project, is_subset, aiModelSettings)
    else:
        # worker not needed
        print("[{}] Subset {} requested, but only {} chunk(s) provided. Skipping...".format(
            project,
            index, len(data)
        ))
        return 0


@current_app.task(name='AIWorker.call_average_model_states', rate_limit=1)
def call_average_model_states(blank, epoch, numEpochs, project, aiModelSettings):
    return worker.call_average_model_states(epoch, numEpochs, project, aiModelSettings)


@current_app.task(name='AIWorker.call_inference')
def call_inference(data, index, epoch, numEpochs, project, aiModelSettings=None, alCriterionSettings=None):
    if len(data) == 2 and data[1] is None:
        # model update call preceded inference task; ignore empty output of it
        data = data[0]

    if index < len(data):
        return worker.call_inference(data[index], epoch, numEpochs, project, aiModelSettings, alCriterionSettings)
    else:
        # worker not needed
        print("[{}] Subset {} requested, but only {} chunk(s) provided. Skipping...".format(
            project,
            index, len(data)
        ))
        return 0


@current_app.task(name='AIWorker.verify_model_state')
def verify_model_state(project):
    return worker.verify_model_state(project)