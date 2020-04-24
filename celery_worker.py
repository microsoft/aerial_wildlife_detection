'''
    Sets up the appropriate Celery consumer(s):
    - AIController
    - AIWorker
    - DataManagement

    depending on the "AIDE_MODULES" environment
    variable.

    2020 Benjamin Kellenberger
'''

import os
from celery import Celery
from kombu import Queue
from kombu.common import Broadcast
from util.configDef import Config

# force enable passive mode
os.environ['PASSIVE_MODE'] = '1'

# parse system config
if not 'AIDE_CONFIG_PATH' in os.environ:
    raise ValueError('Missing system environment variable "AIDE_CONFIG_PATH".')
if not 'AIDE_MODULES' in os.environ:
    raise ValueError('Missing system environment variable "AIDE_MODULES".')
config = Config()


app = Celery('AIDE',
            broker=config.getProperty('AIController', 'broker_URL'),        #TODO
            backend=config.getProperty('AIController', 'result_backend'))   #TODO
app.conf.update(
    result_backend=config.getProperty('AIController', 'result_backend'),    #TODO
    task_ignore_result=False,
    result_persistent=True,
    accept_content = ['json'],
    task_serializer = 'json',
    result_serializer = 'json',
    task_track_started = True,
    broker_pool_limit=None,                 # required to avoid peer connection resets
    broker_heartbeat = 0,                   # required to avoid peer connection resets
    worker_max_tasks_per_child = 1,         # required to free memory (also CUDA) after each process
    task_default_rate_limit = 3,            #TODO
    worker_prefetch_multiplier = 1,         #TODO
    task_acks_late = True,
    task_create_missing_queues = True,
    task_queues = (Broadcast('aide_broadcast'), Queue('FileServer'), Queue('AIController'), Queue('AIWorker')),
    task_routes = {
        'AIController.get_training_images': {
            'queue': 'AIController',
            'routing_key': 'get_training_images'
        },
        'AIController.get_inference_images': {
            'queue': 'AIController',
            'routing_key': 'get_inference_images'
        },
        'AIWorker.call_train': {
            'queue': 'AIWorker',
            'routing_key': 'call_train'
        },
        'AIWorker.call_average_model_states': {
            'queue': 'AIWorker',
            'routing_key': 'call_average_model_states'
        },
        'AIWorker.call_inference': {
            'queue': 'AIWorker',
            'routing_key': 'call_inference'
        },
        'DataAdministration.list_images': {
            'queue': 'FileServer',
            'routing_key': 'list_images'
        },
        'DataAdministration.scan_for_images': {
            'queue': 'FileServer',
            'routing_key': 'scan_for_images'
        },
        'DataAdministration.add_existing_images': {
            'queue': 'FileServer',
            'routing_key': 'add_existing_images'
        },
        'DataAdministration.remove_images': {
            'queue': 'FileServer',
            'routing_key': 'remove_images'
        },
        'DataAdministration.prepare_data_download': {
            'queue': 'FileServer',
            'routing_key': 'prepare_data_download'
        }
    }
    #task_default_queue = Broadcast('aide_admin')
)


# initialize appropriate consumer functionalities
aide_modules = os.environ['AIDE_MODULES'].split(',')
aide_modules = set([a.strip().lower() for a in aide_modules])
num_modules = 0
if 'aicontroller' in aide_modules:
    from modules.AIController.backend import celery_interface as aic_int
    # aic_int.aide_internal_notify({'task': 'add_projects'})
    num_modules += 1
if 'aiworker' in aide_modules:
    from modules.AIWorker.backend import celery_interface as aiw_int
    aiw_int.aide_internal_notify({'task': 'add_projects'})
    num_modules += 1
if 'fileserver' in aide_modules:
    from modules.DataAdministration.backend import celery_interface as da_int
    da_int.aide_internal_notify({'task': 'add_projects'})
    num_modules += 1



if __name__ == '__main__':
    # launch Celery consumer
    if num_modules:
        app.start()