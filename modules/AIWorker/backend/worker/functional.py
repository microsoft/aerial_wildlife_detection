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

    2019-20 Benjamin Kellenberger
'''

from celery import current_task, states
import psycopg2
from psycopg2 import sql
from util.helpers import current_time
from constants.dbFieldNames import FieldNames_annotation, FieldNames_prediction



def __get_message_fun(project):
    def __on_message(state, message, done=None, total=None):
        meta = {
            'project': project
        }
        if (isinstance(done, int) or isinstance(done, float)) and \
            (isinstance(total, int) or isinstance(total, float)):
            meta['done'] = min(done, total)
            meta['total'] = max(done, total)
        if isinstance(message, str):
            meta['message'] = message
        current_task.update_state(
            state=state,
            meta=meta
        )
    return __on_message


def __load_model_state(project, dbConnector):
    # load model state from database
    queryStr = sql.SQL('''
        SELECT query.statedict, query.id FROM (
            SELECT statedict, id, timecreated
            FROM {}
            ORDER BY timecreated DESC NULLS LAST
            LIMIT 1
        ) AS query;
    ''').format(sql.Identifier(project, 'cnnstate'))
    result = dbConnector.execute(queryStr, None, numReturn=1)     #TODO: issues Celery warning if no state dict found
    if not len(result):
        # force creation of new model
        stateDict = None
        stateDictID = None

    else:
        # extract
        stateDict = result[0]['statedict']
        stateDictID = result[0]['id']

    return stateDict, stateDictID



def __load_metadata(project, dbConnector, imageIDs, loadAnnotations):

    # prepare
    meta = {}

    # label names
    labels = {}
    queryStr = sql.SQL(
        'SELECT * FROM {};').format(sql.Identifier(project, 'labelclass'))
    result = dbConnector.execute(queryStr, None, 'all')
    if len(result):
        for r in result:
            labels[r['id']] = r
    meta['labelClasses'] = labels

    # image data
    imageMeta = {}
    if len(imageIDs):
        queryStr = sql.SQL(
            'SELECT * FROM {} WHERE id IN %s').format(sql.Identifier(project, 'image'))
        result = dbConnector.execute(queryStr, (tuple(imageIDs),), 'all')
        if len(result):
            for r in result:
                imageMeta[r['id']] = r

    # annotations
    if loadAnnotations and len(imageIDs):
        # get project's annotation type
        result = dbConnector.execute(sql.SQL('''
                SELECT annotationType
                FROM aide_admin.project
                WHERE shortname = %s;
            '''),
            (project,),
            1)
        annoType = result['annotationtype']

        fieldNames = list(getattr(FieldNames_annotation, annoType).value)
        queryStr = sql.SQL('''
            SELECT id AS annotationID, image, {fieldNames} FROM {id_anno} AS anno
            WHERE image IN %s;
        ''').format(
            fieldNames=sql.SQL(', ').join(fieldNames),
            id_anno=sql.Identifier(project, 'annotation'))
        result = dbConnector.execute(queryStr, (tuple(imageIDs),), 'all')
        if len(result):
            for r in result:
                if not 'annotations' in imageMeta[r['image']]:
                    imageMeta[r['image']]['annotations'] = []
                imageMeta[r['image']]['annotations'].append(r)
    meta['images'] = imageMeta

    return meta


def _call_train(project, imageIDs, subset, trainingFun, dbConnector, fileServer):
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

    print('Initiated training...')
    update_state = __get_message_fun(project)


    # load model state
    update_state(state='PREPARING', message='loading model state')
    try:
        stateDict, _ = __load_model_state(project, dbConnector)
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')


    # load labels and other metadata
    update_state(state='PREPARING', message='loading metadata')
    try:
        data = __load_metadata(project, dbConnector, imageIDs, True)
    except Exception as e:
        print(e)
        raise Exception('error during metadata loading')

    # call training function
    try:
        update_state(state='PREPARING', message='initiating training')
        stateDict = trainingFun(stateDict=stateDict, data=data, updateStateFun=update_state)
    except Exception as e:
        print(e)
        raise Exception('error during training')


    # commit state dict to database
    try:
        update_state(state='FINALIZING', message='saving model state')
        queryStr = sql.SQL('''
            INSERT INTO {} (stateDict, partial)
            VALUES( %s, %s )
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.execute(queryStr, (psycopg2.Binary(stateDict), subset,), numReturn=None)
    except Exception as e:
        print(e)
        raise Exception('error during data committing')

    update_state(state=states.SUCCESS, message='trained on {} images'.format(len(imageIDs)))

    print('Training completed successfully.')
    return 0



def _call_average_model_states(project, averageFun, dbConnector, fileServer):
    '''
        Receives a number of model states (coming from different AIWorker instances),
        averages them by calling the AI model's 'average_model_states' function and inserts
        the returning averaged model state into the database.
    '''

    print('Initiated epoch averaging...')
    update_state = __get_message_fun(project)

    # get all model states
    update_state(state='PREPARING', message='loading model states')
    try:
        queryStr = sql.SQL('''
            SELECT stateDict, model_library FROM {} WHERE partial IS TRUE;
        ''').format(sql.Identifier(project, 'cnnstate'))
        modelStates = dbConnector.execute(queryStr, None, 'all')
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')

    if not len(modelStates):
        # no states to be averaged; return
        update_state(state=states.SUCCESS, message='no model states to be averaged')
        return 0

    # do the work
    update_state(state='PREPARING', message='averaging models')
    try:
        modelStates_avg = averageFun(stateDicts=modelStates)
    except Exception as e:
        print(e)
        raise Exception('error during model state fusion')

    # push to database
    update_state(state='FINALIZING', message='saving model state')
    try:
        model_library = modelStates[0]['model_library']
    except:
        model_library = None
    try:
        queryStr = sql.SQL('''
            INSERT INTO {} (stateDict, partial, model_library)
            VALUES ( %s )
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.insert(queryStr, (modelStates_avg, False, model_library))
    except Exception as e:
        print(e)
        raise Exception('error during data committing')

    # delete partial model states
    update_state(state='FINALIZING', message='purging cache')
    try:
        queryStr = sql.SQL('''
            DELETE FROM {} WHERE partial IS TRUE;
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.execute(queryStr, None, None)
    except Exception as e:
        print(e)
        raise Exception('error during cache purging')

    # all done
    update_state(state=states.SUCCESS, message='averaged {} model states'.format(len(modelStates)))

    print('Model averaging completed successfully.')
    return 0



def _call_inference(project, imageIDs, inferenceFun, rankFun, dbConnector, fileServer):
    '''

    '''
    print('Initiated inference on {} images...'.format(len(imageIDs)))
    update_state = __get_message_fun(project)

    # load model state
    update_state(state='PREPARING', message='loading model state')
    try:
        stateDict, stateDictID = __load_model_state(project, dbConnector)
    except Exception as e:
        print(e)
        raise Exception('error during model state loading')

    # load remaining data (image filenames, class definitions)
    update_state(state='PREPARING', message='loading metadata')
    try:
        data = __load_metadata(project, dbConnector, imageIDs, False)
    except Exception as e:
        print(e)
        raise Exception('error during metadata loading')

    # call inference function
    update_state(state='PREPARING', message='starting inference')
    try:
        result = inferenceFun(stateDict=stateDict, data=data, updateStateFun=update_state)
    except Exception as e:
        print(e)
        raise Exception('error during inference')

    # call ranking function (AL criterion)
    if rankFun is not None:
        update_state(state='PREPARING', message='calculating priorities')
        try:
            result = rankFun(data=result, updateStateFun=update_state, **{'stateDict':stateDict})
        except Exception as e:
            print(e)
            raise Exception('error during ranking')

    # parse result
    try:
        # get project's prediction type
        result = dbConnector.execute(sql.SQL('''
                SELECT predictionType
                FROM aide_admin.project
                WHERE shortname = %s;
            '''),
            (project,),
            1)
        predType = result['predictiontype']

        update_state(state='FINALIZING', message='saving predictions')
        fieldNames = list(getattr(FieldNames_prediction, predType).value)
        fieldNames.append('image')      # image ID
        fieldNames.append('cnnstate')   # model state ID
        values_pred = []
        values_img = []     # mostly for feature vectors
        ids_img = []        # to delete previous predictions
        for imgID in result.keys():
            for prediction in result[imgID]['predictions']:
                nextResultValues = []
                # we expect a dict of values, so we can use the fieldNames directly
                for fn in fieldNames:
                    if fn == 'image':
                        nextResultValues.append(imgID)
                        ids_img.append(imgID)
                    elif fn == 'cnnstate':
                        nextResultValues.append(stateDictID)
                    else:
                        if fn in prediction:
                            #TODO: might need to do typecasts (e.g. UUID?)
                            nextResultValues.append(prediction[fn])

                        else:
                            # field name is not in return value; might need to raise a warning, Exception, or set to None
                            nextResultValues.append(None)
                        
                values_pred.append(tuple(nextResultValues))

            if 'fVec' in result[imgID] and len(result[imgID]['fVec']):
                values_img.append((imgID, psycopg2.Binary(result[imgID]['fVec']),))
    except Exception as e:
        print(e)
        raise Exception('error during result parsing')


    # commit to database
    try:
        if len(values_pred):
            # remove previous predictions first
            queryStr = sql.SQL('''
                DELETE FROM {} WHERE image IN %s;
            ''').format(sql.Identifier(project, 'prediction'))
            dbConnector.insert(queryStr, (ids_img,))
            
            queryStr = sql.SQL('''
                INSERT INTO {id_pred} ( {fieldNames} )
                VALUES %s;
            ''').format(
                id_pred=sql.Identifier(project, 'prediction'),
                fieldNames=sql.SQL(',').join([sql.SQL(f) for f in fieldNames]))
            dbConnector.insert(queryStr, values_pred)

        if len(values_img):
            queryStr = sql.SQL('''
                INSERT INTO {} ( id, fVec )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET fVec = EXCLUDED.fVec;
            ''').format(sql.Identifier(project, 'image'))
            dbConnector.insert(queryStr, values_img)
    except Exception as e:
        print(e)
        raise Exception('error during data committing')
    
    update_state(state=states.SUCCESS, message='predicted on {} images'.format(len(imageIDs)))

    print('Inference completed successfully.')
    return 0