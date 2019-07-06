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



# load AIWorker
worker = AIWorker(config, None)     #TODO: unneccessary second parameter for runserver compatibility



@app.task()
def call_train(data):
    return worker.call_train(data)


@app.task()
def call_average_epochs(modelStates):
    return worker.call_average_epochs(modelStates)


@app.task()
def call_inference(imageIDs):
    return worker.call_inference(imageIDs)



if __name__ == '__main__':
    app.start()