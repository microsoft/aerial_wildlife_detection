'''
    Wrapper for the Celery message broker.

    2019 Benjamin Kellenberger
'''


from celery import Celery
from celery import signals
from celery.bin import Option
from util.configDef import Config

#TODO
from .functional import multiply


#TODO: parse settings.ini file (cf. http://docs.celeryproject.org/en/latest/userguide/extending.html#preload-options)
app = Celery('taskSchedulers', backend='rpc://', broker='pyamqp://guest@localhost//')

@app.task
def task( x, y):
    return multiply(x, y)