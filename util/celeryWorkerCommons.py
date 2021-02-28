'''
    Common functionalities for Celery workers
    (AIController, AIWorker, FileServer).

    2020-21 Benjamin Kellenberger
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



def getCeleryWorkerDetails():
        '''
            Queries all Celery workers for their details (name,
            URL, capabilities, AIDE version, etc.)
        '''
        result = {}
        
        i = current_app.control.inspect()
        workers = i.stats()

        if workers is None or not len(workers):
            return result

        for w in workers:
            aiwV = get_worker_details.s()
            try:
                res = aiwV.apply_async(queue=w)
                res = res.get(timeout=20)                   #TODO: timeout (in seconds)
                if res is None:
                    raise Exception('connection timeout')
                result[w] = res
                result[w]['online'] = True
            except Exception as e:
                result[w] = {
                    'online': False,
                    'message': str(e)
                }
        return result