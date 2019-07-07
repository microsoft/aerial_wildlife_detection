'''
    Routines responsible for the AI model update:
        1. Assemble data and organize
        2. Call the model's anonymous function (train, inference, rank)
        3. Collect the results, assemble, push to database and return
           status

    Also takes care of exceptions and errors, so that the model functions
    don't have to do it.

    The routines defined in this class may be called by multiple modules in
    different ways:
    1. In the first step, an 'AIController' instance receiving a request
       forwards it via a distributed task queue (e.g. Celery).
    2. The task queue then delegates it to one or more connected consumers
       (resp. 'AIWorker' instances).
    3. Each AIWorker instance has a task queue server running and listens to
       jobs distributed by the AIController instance. Whenever a job comes in,
       it calls the very same function the AIController instance delegated and
       processes it (functions below).

    2019 Benjamin Kellenberger
'''

from celery import current_task


def __load_model_state(config, dbConnector):
    # load model state from database
    try:
        sql = '''
            SELECT query.statedict FROM (
                SELECT statedict, timecreated
                FROM {schema}.cnnstate
                ORDER BY timecreated DESC NULLS LAST
                LIMIT 1
            ) AS query;
        '''.format(schema=config.getProperty('Database', 'schema'))
        stateDict = dbConnector.execute(sql, None, numReturn=1)     #TODO: issues Celery warning if no state dict found
    
    except:
        # no state dict in database yet, have to start with a fresh model
        stateDict = None

    return stateDict


def _call_train(dbConnector, config, data, trainingFun, fileServer):
    '''
        Initiates model training and maintains workers, status and failure
        events.

        Inputs:
        - data: a dictionary of the data that got labeled. May contain the following key-value pairs:
                - image ID: { dict }
                    - features: (byte array of feature vectors, if available)
                    - annotations: { dict of annotations as specified in the project settings }
        
        Function then performs sanity checks and forwards the data to the AI model's anonymous
        'train' function, together with some helper instances (a 'Database' instance as well as a
        'FileServer' instance TODO for the model to access more data, if needed).

        Returns:
        - modelStateDict: a new, updated state dictionary of the model as returned by the AI model's
                          'train' function.
        - TODO: more?
    '''

    # 1. Sanity checks, 2. Load model state, 3. Call AI model's train function, 4. Collect results and return
    print('initiate training')

    
    # load model state
    stateDict = __load_model_state(config, dbConnector)

    # load labels and other metadata
    #TODO: query DB based on image IDs received here (in the data object)


    # call training function
    try:
        result = trainingFun(stateDict, data)

    except Exception as err:
        print(err)
        result = 0      #TODO


    #TODO    
    import random
    import time
    time.sleep(5*random.random())
    current_task.update_state('TRAINING', meta={'done': 5,'total': 10})
    time.sleep(5*random.random())

    return result



def _call_average_model_states(dbConnector, config, modelStates, averageFun, fileServer):
    '''
        Receives a number of model states (coming from different AIWorker instances),
        averages them by calling the AI model's 'average_model_states' function and inserts
        the returning averaged model state into the database.
    '''

    #TODO: sanity checks?
    print('initiate epoch averaging')

    # do the work
    modelStates_avg = averageFun(modelStates)

    # push to database
    sql = '''
        INSERT INTO {schema}.cnnstate (stateDict)
        VALUES ( %s )
    '''.format(schema=config.getProperty('Database', 'schema'))     #TODO: multiple CNN types?

    dbConnector.insert(sql, (modelStates_avg,))

    # all done
    return 0



def _call_inference(dbConnector, config, imageIDs, inferenceFun, fileServer):
    '''

    '''

    print('initiated inference on {} images'.format(len(imageIDs)))


    # load model state
    stateDict = __load_model_state(config, dbConnector)


    # call inference function
    result = inferenceFun(stateDict, imageIDs)


    # parse result
    fieldNames = ['label', 'x', 'confidence']     #TODO: get from general helper function
    values = []
    for r in result:
        nextResultValues = []
        # we expect a dict of values, so we can use the fieldNames directly
        for fn in fieldNames:
            if not fn in r:
                # field name is not in return value; might need to raise a warning, Exception, or set to None
                nextResultValues.append(None)
            
            else:
                #TODO: might need to do typecasts (e.g. UUID?)
                nextResultValues.append(r[fn])
        values.append(nextResultValues)


    # commit to database
    sql = '''
        INSERT INTO {schema}.prediction ( {fieldNames} )
        VALUES %s;
    '''.format(schema=config.getProperty('Database', 'schema'),
        fieldNames=fieldNames)

    dbConnector.insert(sql, tuple(values))


    #TODO: return status?
    return 0



def _call_rank(dbConnector, config, data, rankFun, fileServer):
    '''

    '''
    print('Active Learning criterion.')

    return 0