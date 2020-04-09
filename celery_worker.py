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
from kombu.common import Broadcast
from util.configDef import Config


# parse system config
if not 'AIDE_CONFIG_PATH' in os.environ:
    raise ValueError('Missing system environment variable "AIDE_CONFIG_PATH".')
if not 'AIDE_MODULES' in os.environ:
    raise ValueError('Missing system environment variable "AIDE_MODULES".')
config = Config()


aide_modules = os.environ['AIDE_MODULES'].split(',')
aide_modules = set([a.strip().lower() for a in aide_modules])


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
    broker_heartbeat = 0,           # required to avoid peer connection resets
    worker_max_tasks_per_child = 1,      # required to free memory (also CUDA) after each process
    task_default_rate_limit = 3,         #TODO
    worker_prefetch_multiplier = 1,          #TODO
    task_acks_late = True,
    task_create_missing_queues = True,
    task_queues = (Broadcast('aide_broadcast'),),
    task_routes = {
        'aide_admin': {
            'queue': 'aide_broadcast',
            'exchange': 'aide_broadcast'
        }                                       #TODO: separate queue for data management and AIController
    }
    #task_default_queue = Broadcast('aide_admin')
)


# initialize appropriate consumer functionalities
num_modules = 0
#TODO
if 'aicontroller' in aide_modules:
    from modules.AIController.backend import celery_interface as aic_int
    num_modules += 1
if 'aiworker' in aide_modules:
    from modules.AIWorker.backend import celery_interface as aiw_int
    num_modules += 1
# if 'aicontroller' in aide_modules or 'aiworker' in aide_modules:
#     from modules.AIWorker.backend import celery_interface as ai_int
#     num_modules += 1

if 'fileserver' in aide_modules:
    from modules.DataAdministration.backend import celery_interface as da_int
    num_modules += 1



if __name__ == '__main__':
    # launch Celery consumer
    if num_modules:
        app.start()