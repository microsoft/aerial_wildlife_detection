'''
    Wrapper for the Celery message broker concerning
    the AIWorker(s).

    2019-20 Benjamin Kellenberger
'''

from celery import current_app
from kombu.common import Broadcast
from modules.AIWorker.app import AIWorker
from util.configDef import Config


# init AIWorker
worker = AIWorker(Config(), None)


@current_app.task()
def aide_internal_notify(message):
    return worker.aide_internal_notify(message)


@current_app.task(rate_limit=1)
def call_train(project, data, subset):
    return worker.call_train(project, data, subset)


@current_app.task(rate_limit=1)
def call_average_model_states(project, *args):
    return worker.call_average_model_states(project)


@current_app.task()
def call_inference(project, imageIDs):
    return worker.call_inference(project, imageIDs)