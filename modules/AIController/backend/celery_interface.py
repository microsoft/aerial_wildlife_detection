'''
    Wrapper for the Celery message broker.

    2019 Benjamin Kellenberger
'''

import os
from configparser import ConfigParser
from celery import Celery
from celery import signals
from celery.bin import Option

from modules.AIWorker.app import AIWorker

from util.configDef import Config



# parse system config
if not 'AIDE_CONFIG_PATH' in os.environ:
    raise ValueError('Missing system environment variable "AIDE_CONFIG_PATH".')
config = Config()

app = Celery('AIController',
            broker=config.getProperty('AIController', 'broker_URL'),
            backend=config.getProperty('AIController', 'result_backend'))
app.conf.update(
    result_backend=config.getProperty('AIController', 'result_backend'),
    task_ignore_result=False,
    result_persistent=True,
    accept_content = ['json'],
    task_serializer = 'json',
    result_serializer = 'json',
    task_track_started = True,
    broker_heartbeat = 0,           # required to avoid peer connection resets
    worker_max_tasks_per_child = 1,      # required to free memory (also CUDA) after each process
    task_default_rate_limit = 3         #TODO
)


# load AIWorker     TODO: since this file gets imported through the AIController as well, it has to unnecessarily create an AIWorker instance...
worker = AIWorker(config, None)     #TODO: unneccessary second parameter for runserver compatibility





@app.task(rate_limit=1)
def call_train(data, subset):
    return worker.call_train(data, subset)


@app.task(rate_limit=1)
def call_average_model_states(*args):
    return worker.call_average_model_states()


@app.task()
def call_inference(imageIDs):
    return worker.call_inference(imageIDs)



if __name__ == '__main__':
    app.start()