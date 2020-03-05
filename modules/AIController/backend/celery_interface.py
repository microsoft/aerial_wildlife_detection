'''
    Wrapper for the Celery message broker concerning
    the AIController and AIWorkers.

    2019-20 Benjamin Kellenberger
'''

import os
from configparser import ConfigParser
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
    # worker = _get_worker()
    return worker.call_train(project, data, subset)


@current_app.task(rate_limit=1)
def call_average_model_states(project, *args):
    # worker = _get_worker()
    return worker.call_average_model_states(project)


@current_app.task()
def call_inference(project, imageIDs):
    # worker = _get_worker()
    return worker.call_inference(project, imageIDs)



# # parse system config
# if not 'AIDE_CONFIG_PATH' in os.environ:
#     raise ValueError('Missing system environment variable "AIDE_CONFIG_PATH".')
# config = Config()
# if config.getProperty('Project', 'demoMode', type=bool, fallback=False):
#     raise Exception('AIController and AIWorkers cannot be launched in demo mode.')

# app = Celery('AIController',
#             broker=config.getProperty('AIController', 'broker_URL'),
#             backend=config.getProperty('AIController', 'result_backend'))
# app.conf.update(
#     result_backend=config.getProperty('AIController', 'result_backend'),
#     task_ignore_result=False,
#     result_persistent=True,
#     accept_content = ['json'],
#     task_serializer = 'json',
#     result_serializer = 'json',
#     task_track_started = True,
#     broker_heartbeat = 0,           # required to avoid peer connection resets
#     worker_max_tasks_per_child = 1,      # required to free memory (also CUDA) after each process
#     task_default_rate_limit = 3,         #TODO
#     worker_prefetch_multiplier = 1,          #TODO
#     task_acks_late = True,
#     task_create_missing_queues = True,
#     task_queues = (Broadcast('aide_broadcast'),),
#     task_routes = {
#         'aide_admin': {
#             'queue': 'aide_broadcast',
#             'exchange': 'aide_broadcast'
#         }
#     }
#     #task_default_queue = Broadcast('aide_admin')
# )


# # init AIWorker
# worker = AIWorker(config, None)


# @app.task()
# def aide_internal_notify(message):
#     return worker.aide_internal_notify(message)


# @app.task(rate_limit=1)
# def call_train(project, data, subset):
#     # worker = _get_worker()
#     return worker.call_train(project, data, subset)


# @app.task(rate_limit=1)
# def call_average_model_states(project, *args):
#     # worker = _get_worker()
#     return worker.call_average_model_states(project)


# @app.task()
# def call_inference(project, imageIDs):
#     # worker = _get_worker()
#     return worker.call_inference(project, imageIDs)



# if __name__ == '__main__':
#     app.start()