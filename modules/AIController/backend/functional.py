'''
    Helper routines to assemble, query and commit data to and from the
    database for the AIController.

    2019 Benjamin Kellenberger
'''

#TODO: test:
def multiply(a, b):
    print('it still works!')
    return a * b



#TODO: move to AIWorker?
#
# Idea:
# - AIController simply accepts requests, queries DB, prepares data
#   and then launches workers through Celery
# - Workers contain the actual function that then calls the ai.nn
#   model(s).
#
# So, the caller functions (like here) belong to the AIWorker, but
# the Celery wrapper (cf. celery.py) should be in the AIController.
# This effectively bridges the two units and ties them together.
def _call_train():
    pass