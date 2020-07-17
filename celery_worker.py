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
import celery
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


# parse AIDE modules and set up queues
queues = []
aideModules = os.environ['AIDE_MODULES'].split(',')
aideModules = set([a.strip().lower() for a in aideModules])
for m in aideModules:
    module = m.strip().lower()
    if module == 'aicontroller':
        queues.append(Queue('AIController'))
    elif module == 'fileserver':
        queues.append(Queue('FileServer'))
    elif module == 'aiworker':
        queues.append(Queue('AIWorker'))

queues.extend([
    Broadcast('aide_broadcast'),
    Queue('aide@'+celery.utils.nodenames.gethostname())
])



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
    task_queues = tuple(queues),
    task_routes = {
        'general.get_worker_details': {
            'queue': 'aide@'+celery.utils.nodenames.gethostname(),
            'routing_key': 'worker_details'
        },
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
        },
        'DataAdministration.watch_image_folders': {
            'queue': 'FileServer',
            'routing_key': 'watch_image_folders'
        }
    }
    #task_default_queue = Broadcast('aide_admin')
)


# initialize appropriate consumer functionalities
from util import celeryWorkerCommons

num_modules = 0
if 'aicontroller' in aideModules:
    from modules.AIController.backend import celery_interface as aic_int
    num_modules += 1
if 'aiworker' in aideModules:
    from modules.AIWorker.backend import celery_interface as aiw_int
    num_modules += 1
if 'fileserver' in aideModules:
    from modules.DataAdministration.backend import celery_interface as da_int
    num_modules += 1

    # scanning project folders for new images: set up periodic task
    scanInterval = config.getProperty('FileServer', 'watch_folder_interval', type=float, fallback=60)
    if scanInterval > 0:
        @app.on_after_configure.connect
        def setup_periodic_tasks(sender, **kwargs):
            sender.add_periodic_task(scanInterval, da_int.watchImageFolders.s())



if __name__ == '__main__':
    # launch Celery consumer
    if num_modules:
        app.start()