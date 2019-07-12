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

from celery import current_task, states
import psycopg2
from util.helpers import current_time
from constants.dbFieldNames import FieldNames_annotation, FieldNames_prediction


def __load_model_state(config, dbConnector):
    # load model state from database
    sql = '''
        SELECT query.statedict FROM (
            SELECT statedict, timecreated
            FROM {schema}.cnnstate
            ORDER BY timecreated DESC NULLS LAST
            LIMIT 1
        ) AS query;
    '''.format(schema=config.getProperty('Database', 'schema'))
    stateDict = dbConnector.execute(sql, None, numReturn=1)     #TODO: issues Celery warning if no state dict found
    if not len(stateDict):
        # force creation of new model
        stateDict = None
    
    else:
        # extract
        stateDict = stateDict[0]['statedict']

    return stateDict


def __load_metadata(config, dbConnector, imageIDs, loadAnnotations):

    schema = config.getProperty('Database', 'schema')

    # prepare
    meta = {}

    # label names
    labels = {}
    sql = 'SELECT * FROM {schema}.labelclass;'.format(schema=schema)
    result = dbConnector.execute(sql, None, 'all')
    for r in result:
        labels[r['id']] = r     #TODO: make more elegant?
    meta['labelClasses'] = labels

    # image data
    imageMeta = {}
    sql = 'SELECT * FROM {schema}.image WHERE id IN %s'.format(schema=schema)
    result = dbConnector.execute(sql, (tuple(imageIDs),), 'all')
    for r in result:
        imageMeta[r['id']] = r  #TODO: make more elegant?


    # annotations
    if loadAnnotations:
        fieldNames = list(getattr(FieldNames_annotation, config.getProperty('Project', 'predictionType')).value)
        sql = '''
            SELECT id AS annotationID, image, {fieldNames} FROM {schema}.annotation AS anno
            WHERE image IN %s;
        '''.format(schema=schema, fieldNames=','.join(fieldNames))
        result = dbConnector.execute(sql, (tuple(imageIDs),), 'all')
        for r in result:
            if not 'annotations' in imageMeta[r['image']]:
                imageMeta[r['image']]['annotations'] = []
            imageMeta[r['image']]['annotations'].append(r)      #TODO: make more elegant?
    meta['images'] = imageMeta

    return meta


def _call_train(dbConnector, config, imageIDs, subset, trainingFun, fileServer):
    '''
        Initiates model training and maintains workers, status and failure
        events.

        Inputs:
        - imageIDs: a list of image UUIDs the model should be trained on. Note that the remaining
                    metadata (labels, class definitions, etc.) will be loaded here.
        
        Function then performs sanity checks and forwards the data to the AI model's anonymous
        'train' function, together with some helper instances (a 'Database' instance as well as a
        'FileServer' instance TODO for the model to access more data, if needed).

        Returns:
        - modelStateDict: a new, updated state dictionary of the model as returned by the AI model's
                          'train' function.
        - TODO: more?
    '''

    #TODO
    print('initiate training')

    
    # load model state
    current_task.update_state(state='PREPARING', meta={'message':'loading model state'})
    try:
        stateDict = __load_model_state(config, dbConnector)
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')


    # load labels and other metadata
    current_task.update_state(state='PREPARING', meta={'message':'loading metadata'})
    try:
        data = __load_metadata(config, dbConnector, imageIDs, True)
    except Exception as e:
        print(e)
        raise Exception('error during metadata loading')

    # call training function
    try:
        current_task.update_state(state='PREPARING', meta={'message':'initiating training'})
        stateDict = trainingFun(stateDict=stateDict, data=data)
    except Exception as e:
        print(e)
        raise Exception('error during training')

    #TODO
    return 0

    # commit state dict to database
    try:
        current_task.update_state(state='FINALIZING', meta={'message':'saving model state'})
        sql = '''
            INSERT INTO {schema}.cnnstate(stateDict, partial)
            VALUES( %s, %s )
        '''.format(schema=config.getProperty('Database', 'schema'))
        dbConnector.execute(sql, (psycopg2.Binary(stateDict), subset,), numReturn=None)
    except Exception as e:
        print(e)
        raise Exception('error during data committing')

    current_task.update_state(state=states.SUCCESS, meta={'message':'done'})
    return 0



def _call_average_model_states(dbConnector, config, averageFun, fileServer):
    '''
        Receives a number of model states (coming from different AIWorker instances),
        averages them by calling the AI model's 'average_model_states' function and inserts
        the returning averaged model state into the database.
    '''

    #TODO: sanity checks?
    print('initiate epoch averaging')
    schema = config.getProperty('Database', 'schema')

    # get all model states
    current_task.update_state(state='PREPARING', meta={'message':'loading model states'})
    try:
        sql = '''
            SELECT stateDict FROM {schema}.cnnstate WHERE partial IS TRUE;
        '''.format(schema=schema)
        modelStates = dbConnector.execute(sql, None, 'all')
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')

    if not len(modelStates):
        # no states to be averaged; return
        current_task.update_state(state=states.SUCCESS, meta={'message':'no model states to be averaged'})
        return 0


    # do the work
    current_task.update_state(state='PREPARING', meta={'message':'averaging models'})
    try:
        modelStates_avg = averageFun(stateDicts=modelStates)
    except Exception as e:
        print(e)
        raise Exception('error during model state fusion')

    # push to database
    current_task.update_state(state='FINALIZING', meta={'message':'saving model state'})
    try:
        sql = '''
            INSERT INTO {schema}.cnnstate (stateDict, partial)
            VALUES ( %s )
        '''.format(schema=schema)     #TODO: multiple CNN types?
        dbConnector.insert(sql, (modelStates_avg, False,))   #TODO
    except Exception as e:
        print(e)
        raise Exception('error during data committing')


    # delete partial model states
    current_task.update_state(state='FINALIZING', meta={'message':'purging cache'})
    try:
        sql = '''
            DELETE FROM {schema}.cnnstate WHERE partial IS TRUE;
        '''.format(schema=schema)
        dbConnector.execute(sql, None, None)
    except Exception as e:
        print(e)
        raise Exception('error during cache purging')


    # all done
    current_task.update_state(state=states.SUCCESS, meta={'message':'done'})
    return 0



def _call_inference(dbConnector, config, imageIDs, inferenceFun, rankFun, fileServer):
    '''

    '''

    print('initiated inference on {} images'.format(len(imageIDs)))


    # load model state
    current_task.update_state(state='PREPARING', meta={'message':'loading model state'})
    try:
        stateDict = __load_model_state(config, dbConnector)
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')

    # load remaining data (image filenames, class definitions)
    current_task.update_state(state='PREPARING', meta={'message':'loading metadata'})
    try:
        data = __load_metadata(config, dbConnector, imageIDs, False)
    except Exception as e:
        print(e)
        raise Exception('error during metadata loading')

    # call inference function
    current_task.update_state(state='PREPARING', meta={'message':'starting inference'})
    try:
        result = inferenceFun(stateDict=stateDict, data=data)
    except Exception as e:
        print(e)
        raise Exception('error during inference')

    # call ranking function (AL criterion)
    if rankFun is not None:
        current_task.update_state(state='PREPARING', meta={'message':'calculating priorities'})
        try:
            result = rankFun(data=result, **{'stateDict':stateDict})
        except Exception as e:
            print(e)
            raise Exception('error during ranking')

    # parse result
    try:
        current_task.update_state(state='FINALIZING', meta={'message':'saving predictions'})
        fieldNames = list(getattr(FieldNames_prediction, config.getProperty('Project', 'predictionType')).value)
        fieldNames.append('image')  # image ID
        values_pred = []
        values_img = []     # mostly for feature vectors
        for imgID in result.keys():
            for prediction in result[imgID]['predictions']:
                nextResultValues = []
                # we expect a dict of values, so we can use the fieldNames directly
                for fn in fieldNames:
                    if fn == 'image':
                        nextResultValues.append(imgID)
                    else:
                        if fn in prediction:
                            #TODO: might need to do typecasts (e.g. UUID?)
                            nextResultValues.append(prediction[fn])

                        else:
                            # field name is not in return value; might need to raise a warning, Exception, or set to None
                            nextResultValues.append(None)
                        
                values_pred.append(nextResultValues)

            if 'fVec' in result[imgID] and len(result[imgID]['fVec']):
                values_img.append([imgID, result[imgID]['fVec']])
    except Exception as e:
        print(e)
        raise Exception('error during result parsing')

    #TODO
    return 0

    # commit to database
    try:
        if len(values_pred):
            sql = '''
                INSERT INTO {schema}.prediction ( {fieldNames} )
                VALUES %s;
            '''.format(schema=config.getProperty('Database', 'schema'),
                fieldNames=fieldNames)
            dbConnector.insert(sql, tuple(values_pred))

        if len(values_img):
            sql = '''
                INSERT INTO {schema}.image ( id, fVec )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET fVec = EXCLUDED.fVec;
            '''.format(schema=config.getProperty('Database', 'schema'))
            dbConnector.insert(sql, tuple(values_img))
    except Exception as e:
        print(e)
        raise Exception('error during data committing')

    
    current_task.update_state(state=states.SUCCESS, meta={'message':'done'})
    return 0



#TODO: ranking is being done automatically after inference (requires logits which are not stored in DB)
# def _call_rank(dbConnector, config, predictions, rankFun, fileServer):
#     '''

#     '''

#     try:
#         result = rankFun(data=predictions)
#     except Exception as e:
#         print(e)
#         raise Exception('error during ranking')

#     return result