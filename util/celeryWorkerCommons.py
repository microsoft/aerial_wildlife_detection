'''
    Common functionalities for Celery workers
    (AIController, AIWorker, FileServer).

    2020 Benjamin Kellenberger
'''

import os
from celery import current_app
from constants.version import AIDE_VERSION


def _get_modules():
    modules = os.environ['AIDE_MODULES'].split(',')
    modules = set([m.strip().lower() for m in modules])
    return {
        'AIController': ('aicontroller' in modules),
        'AIWorker': ('aiworker' in modules),
        'FileServer': ('fileserver' in modules)
    }



@current_app.task(name='general.get_worker_details')
def get_worker_details():
    # get modules
    return {
        'aide_version': AIDE_VERSION,
        'modules': _get_modules()
        #TODO: GPU capabilities, CPU?
    }