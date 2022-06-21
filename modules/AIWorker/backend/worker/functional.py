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

    2019-22 Benjamin Kellenberger
'''

import base64
import json
import numpy as np
from celery import current_task, states
import psycopg2
from psycopg2 import sql
from util.helpers import DEFAULT_RENDER_CONFIG, array_split, DEFAULT_BAND_CONFIG
from constants.dbFieldNames import FieldNames_annotation, FieldNames_prediction



def __get_message_fun(project, cumulatedTotal=None, offset=0, epoch=None, numEpochs=None):
    def __on_message(state, message, done=None, total=None):
        meta = {
            'project': project,
            'epoch': epoch
        }
        if (isinstance(done, int) or isinstance(done, float)) and \
            (isinstance(total, int) or isinstance(total, float)):
            trueTotal = total + offset
            if isinstance(cumulatedTotal, int) or isinstance(cumulatedTotal, float):
                trueTotal = max(trueTotal, cumulatedTotal)
            meta['done'] = min(done + offset, trueTotal)
            meta['total'] = max(meta['done'], trueTotal)    #max(done, trueTotal)

        message_combined = ''
        if isinstance(epoch, int) and isinstance(numEpochs, int) and numEpochs > 1:
            message_combined += f'[Epoch {epoch}/{numEpochs}] '

        if isinstance(message, str):
            message_combined += message
        
        if len(message_combined):
            meta['message'] = message_combined
        current_task.update_state(
            state=state,
            meta=meta
        )
    return __on_message


def __load_model_state(project, modelLibrary, dbConnector):
    # load model state from database
    queryStr = sql.SQL('''
        SELECT query.statedict, query.id, query.marketplace_origin_id, query.labelclass_autoupdate FROM (
            SELECT statedict, id, timecreated, marketplace_origin_id, labelclass_autoupdate
            FROM {}
            WHERE model_library = %s
            ORDER BY timecreated DESC NULLS LAST
            LIMIT 1
        ) AS query;
    ''').format(sql.Identifier(project, 'cnnstate'))
    result = dbConnector.execute(queryStr, (modelLibrary,), numReturn=1)     #TODO: issues Celery warning if no state dict found
    if result is None or not len(result):
        # force creation of new model
        stateDict = None
        stateDictID = None
        modelOriginID = None
        labelclass_autoupdate = True    # new model = no existing class map

    else:
        # extract
        stateDict = result[0]['statedict']
        stateDictID = result[0]['id']
        modelOriginID = result[0]['marketplace_origin_id']
        labelclass_autoupdate = result[0]['labelclass_autoupdate']

    return stateDict, stateDictID, modelOriginID, labelclass_autoupdate



def __load_metadata(project, dbConnector, imageIDs, loadAnnotations, modelOriginID):

    # prepare
    meta = {}
    if imageIDs is None:
        imageIDs = []

    # label names and model-to-project mapping (if present)
    labels = {}
    queryStr = sql.SQL('''
        SELECT * FROM {id_lc} AS lc
        LEFT OUTER JOIN (
            SELECT * FROM {id_lcmap}
            WHERE marketplace_origin_id = %s
        ) AS lcm
        ON lc.id = lcm.labelclass_id_project
    ''').format(
        id_lc=sql.Identifier(project, 'labelclass'),
        id_lcmap=sql.Identifier(project, 'model_labelclass')
    )
    result = dbConnector.execute(queryStr, (modelOriginID,), 'all')
    if result is not None and len(result):
        for r in result:
            labels[r['id']] = r
    meta['labelClasses'] = labels

    # image format (band configuration)
    #NOTE: we also load the render config for those models that don't specify which bands to take
    # (i.e., we let the model use the same RGB or grayscale bands as used for the visualizer front-end)
    try:
        queryStr = sql.SQL('''
            SELECT band_config, render_config
            FROM aide_admin.project
            WHERE shortname = %s;
        ''')
        config = dbConnector.execute(queryStr, (project,), 1)
        bandConfig = json.loads(config[0]['band_config'])
        if bandConfig is None:
            raise
        try:
            renderConfig = json.loads(config[0]['render_config'])
            if renderConfig is None:
                raise
        except:
            # default render configuration; with adjustment to number of bands
            renderConfig = DEFAULT_RENDER_CONFIG
            renderConfig['bands']['indices'] = {
                'red': 0,
                'green': min(1, len(bandConfig)-1),
                'blue': min(2, len(bandConfig)-1)
            }
    except:
        # fallback to default
        bandConfig = DEFAULT_BAND_CONFIG
        renderConfig = DEFAULT_RENDER_CONFIG
    meta['band_config'] = bandConfig
    meta['render_config'] = renderConfig

    # image data
    imageMeta = {}
    if len(imageIDs):
        queryStr = sql.SQL(
            'SELECT * FROM {} WHERE id IN %s').format(sql.Identifier(project, 'image'))
        result = dbConnector.execute(queryStr, (tuple(imageIDs),), 'all')
        if len(result):
            for r in result:
                # append window to filename for correct loading
                if r.get('x', None) is not None and r.get('y', None) is not None:
                    r['filename'] += '?window={},{},{},{}'.format(
                        r['y'], r['x'], r['height'], r['width']
                    )
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
        annoType = result[0]['annotationtype']

        fieldNames = list(getattr(FieldNames_annotation, annoType).value)
        queryStr = sql.SQL('''
            SELECT id AS annotationID, image, {fieldNames} FROM {id_anno} AS anno
            WHERE image IN %s;
        ''').format(
            fieldNames=sql.SQL(', ').join([sql.SQL(f) for f in fieldNames]),
            id_anno=sql.Identifier(project, 'annotation'))
        result = dbConnector.execute(queryStr, (tuple(imageIDs),), 'all')
        if len(result):
            for r in result:
                if not 'annotations' in imageMeta[r['image']]:
                    imageMeta[r['image']]['annotations'] = []
                imageMeta[r['image']]['annotations'].append(r)
    meta['images'] = imageMeta

    return meta


def __get_ai_library_names(project, dbConnector):
    model_library, alcriterion_library = None, None
    try:
        queryStr = sql.SQL('''
            SELECT ai_model_library, ai_alcriterion_library
            FROM "aide_admin".project
            WHERE shortname = %s;
        ''')
        result = dbConnector.execute(queryStr, (project,), 1)
        model_library = result[0]['ai_model_library']
        alcriterion_library = result[0]['ai_alcriterion_library']
    finally:
        return model_library, alcriterion_library


def _call_update_model(project, numEpochs, modelInstance, modelLibrary, dbConnector):
    '''
        Checks first if any label classes have been added since the last model update.
        If so, or if a new model has been selected, this calls the model update procedure
        that is supposed to modify the model to incorporate newly added label classes.
        Returns the updated model state dict.
    '''
    update_state = __get_message_fun(project, numEpochs=numEpochs)

    # abort if model does not support updating
    if not hasattr(modelInstance, 'update_model'):
        print(f'[{project} - model update] WARNING: model "{modelLibrary}" does not support modification to new label classes. Skipping...')
        update_state(state=states.SUCCESS, message=f'[{project} - model update] WARNING: model does not support modification to new label classes and has not been updated.')
        return

    print(f'[{project}] Updating model to incorporate potentially new label classes...')

    # load model state
    update_state(state='PREPARING', message=f'[{project} - model update] loading model state')
    try:
        stateDict, _, modelOriginID, labelclass_autoupdate = __load_model_state(project, modelLibrary, dbConnector)
    except Exception as e:
        print(e)
        raise Exception(f'[{project} - model update] error during model state loading (reason: {str(e)})')

    # check if explicit model updating is enabled
    autoUpdateEnabled = dbConnector.execute(sql.SQL('''
        SELECT labelclass_autoupdate AS au
        FROM aide_admin.project
        WHERE shortname = %s;
    ''').format(
        id_cnnstate=sql.Identifier(project, 'cnnstate')
    ), (project,), 1)
    autoUpdateEnabled = (autoUpdateEnabled[0]['au'] or labelclass_autoupdate)
    if not autoUpdateEnabled:
        print(f'[{project}] Model auto-update disabled; skipping...')
        return

    # load labels and other metadata
    update_state(state='PREPARING', message=f'[{project} - model update] loading metadata')
    try:
        data = __load_metadata(project, dbConnector, None, True, modelOriginID)
    except Exception as e:
        print(e)
        raise Exception(f'[{project} - model update] error during metadata loading (reason: {str(e)})')

    # call update function
    try:
        update_state(state='PREPARING', message=f'[{project} - model update] initiating model update')
        stateDict = modelInstance.update_model(stateDict=stateDict, data=data, updateStateFun=update_state)
    except Exception as e:
        print(e)
        raise Exception(f'[{project} - model update] error during model update (reason: {str(e)})')


    # commit state dict to database
    try:
        update_state(state='FINALIZING', message=f'[{project} - model update] saving model state')
        model_library, alcriterion_library = __get_ai_library_names(project, dbConnector)
        queryStr = sql.SQL('''
            INSERT INTO {} (stateDict, partial, model_library, alcriterion_library, marketplace_origin_id, labelclass_autoupdate)
            VALUES( %s, %s, %s, %s, %s, TRUE )
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.execute(queryStr, (psycopg2.Binary(stateDict), False, model_library, alcriterion_library, modelOriginID), numReturn=None)
    except Exception as e:
        print(e)
        raise Exception(f'[{project} - model update] error during data committing (reason: {str(e)})')

    update_state(state=states.SUCCESS, message='model updated')
    return



def _call_train(project, imageIDs, epoch, numEpochs, subset, modelInstance, modelLibrary, dbConnector):
    '''
        Initiates model training and maintains workers, status and failure
        events.

        Inputs:
        - imageIDs: a list of image UUIDs the model should be trained on. Note that the remaining
                    metadata (labels, class definitions, etc.) will be loaded here.
        
        Function then performs sanity checks and forwards the data to the AI model's anonymous
        'train' function, together with some helper instances (a 'Database' instance as well as a

        Returns:
        - modelStateDict: a new, updated state dictionary of the model as returned by the AI model's
                          'train' function.
        - TODO: more?
    '''

    print(f'[{project}] Epoch {epoch}: Initiated training...')
    update_state = __get_message_fun(project, len(imageIDs), 0, epoch, numEpochs)


    # load model state
    update_state(state='PREPARING', message=f'[Epoch {epoch}] loading model state')
    try:
        stateDict, _, modelOriginID, labelclass_autoupdate = __load_model_state(project, modelLibrary, dbConnector)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during model state loading (reason: {str(e)})')


    # load labels and other metadata
    update_state(state='PREPARING', message=f'[Epoch {epoch}] loading metadata')
    try:
        data = __load_metadata(project, dbConnector, imageIDs, True, modelOriginID)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during metadata loading (reason: {str(e)})')

    # call training function
    try:
        update_state(state='PREPARING', message=f'[Epoch {epoch}] initiating training')
        result = modelInstance.train(stateDict=stateDict, data=data, updateStateFun=update_state)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during training (reason: {str(e)})')

    # separate model state and statistics (if provided)
    if isinstance(result, tuple):
        stateDict = result[0]
        stats = result[1]
        if isinstance(stats, dict):
            stats = json.dumps(stats)
        else:
            stats = None
    else:
        stateDict = result
        stats = None

    # commit state dict to database
    try:
        update_state(state='FINALIZING', message=f'[Epoch {epoch}] saving model state')
        model_library, alcriterion_library = __get_ai_library_names(project, dbConnector)
        queryStr = sql.SQL('''
            INSERT INTO {} (stateDict, stats, partial, model_library, alcriterion_library, marketplace_origin_id, labelclass_autoupdate)
            VALUES( %s, %s, %s, %s, %s, %s, %s )
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.execute(queryStr, (psycopg2.Binary(stateDict), stats, subset, model_library, alcriterion_library, modelOriginID, labelclass_autoupdate), numReturn=None)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during data committing (reason: {str(e)})')

    update_state(state=states.SUCCESS, message='trained on {} images'.format(len(imageIDs)))

    print(f'[{project}] Epoch {epoch}: Training completed successfully.')
    return



def _call_average_model_states(project, epoch, numEpochs, modelInstance, modelLibrary, dbConnector):
    '''
        Receives a number of model states (coming from different AIWorker instances),
        averages them by calling the AI model's 'average_model_states' function and inserts
        the returning averaged model state into the database.
    '''

    print(f'[{project}] Epoch {epoch}: Initiated model state averaging...')
    update_state = update_state = __get_message_fun(project, None, 0, epoch, numEpochs)

    # get all model states
    update_state(state='PREPARING', message=f'[Epoch {epoch}] loading model states')
    try:
        queryStr = sql.SQL('''
            SELECT stateDict, statistics, model_library, alcriterion_library, marketplace_origin_id
            FROM {}
            WHERE partial IS TRUE AND model_library = %s;
        ''').format(sql.Identifier(project, 'cnnstate'))
        queryResult = dbConnector.execute(queryStr, (modelLibrary,), 'all')
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during model state loading (reason: {str(e)})')

    if not len(queryResult):
        # no states to be averaged; return
        print(f'[{project}] Epoch {epoch}: No model states to be averaged.')
        update_state(state=states.SUCCESS, message=f'[Epoch {epoch}] no model states to be averaged')
        return

    # do the work
    update_state(state='PREPARING', message=f'[Epoch {epoch}] averaging models')
    try:
        modelOriginID = queryResult[0]['marketplace_origin_id']
        modelStates = [qr['statedict'] for qr in queryResult]
        modelStates_avg = modelInstance.average_model_states(stateDicts=modelStates, updateStateFun=update_state)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during model state fusion (reason: {str(e)})')
    
    # average statistics values (if present)
    stats_avg = {}
    dicts = [json.loads(qr['stats']) for qr in queryResult if qr['stats'] is not None]
    keys = [list(d.keys()) for d in dicts]
    keys = set([key for stats in keys for key in stats])
    for key in keys:
        stats_avg[key] = np.nanmean([d[key] for d in dicts if key in d])
    if not len(stats_avg):
        stats_avg = None

    # push to database
    update_state(state='FINALIZING', message=f'[Epoch {epoch}] saving model state')
    try:
        model_library = queryResult[0]['model_library']
        alcriterion_library = queryResult[0]['alcriterion_library']
    except:
        # load model library from database
        model_library, alcriterion_library = __get_ai_library_names(project, dbConnector)
    try:
        queryStr = sql.SQL('''
            INSERT INTO {} (stateDict, stats, partial, model_library, alcriterion_library, marketplace_origin_id)
            VALUES ( %s )
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.insert(queryStr, (modelStates_avg, stats_avg, False, model_library, alcriterion_library, modelOriginID))
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during data committing (reason: {str(e)})')

    # delete partial model states
    update_state(state='FINALIZING', message=f'[Epoch {epoch}] purging cache')
    try:
        queryStr = sql.SQL('''
            DELETE FROM {} WHERE partial IS TRUE;
        ''').format(sql.Identifier(project, 'cnnstate'))
        dbConnector.execute(queryStr, None, None)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during cache purging (reason: {str(e)})')

    # all done
    update_state(state=states.SUCCESS, message=f'[Epoch {epoch}] averaged {len(queryResult)} model states')

    print(f'[{project}] Epoch {epoch}: Model averaging completed successfully.')
    return



def _call_inference(project, imageIDs, epoch, numEpochs, modelInstance, modelLibrary, rankerInstance, dbConnector, batchSizeLimit):
    '''

    '''
    print(f'[{project}] Epoch {epoch}: Initiated inference on {len(imageIDs)} images...')
    update_state = __get_message_fun(project, len(imageIDs), 0, epoch, numEpochs)

    # get project's prediction type
    projectMeta = dbConnector.execute(sql.SQL('''
            SELECT predictionType
            FROM aide_admin.project
            WHERE shortname = %s;
        '''),
        (project,),
        1)
    predType = projectMeta[0]['predictiontype']

    # load model state
    update_state(state='PREPARING', message=f'[Epoch {epoch}] loading model state')
    try:
        stateDict, stateDictID, modelOriginID, _ = __load_model_state(project, modelLibrary, dbConnector)
    except Exception as e:
        print(e)
        raise Exception(f'[Epoch {epoch}] error during model state loading (reason: {str(e)})')

    # if batch size limit specified: split imageIDs into chunks and process in smaller batches
    if isinstance(batchSizeLimit, int) and batchSizeLimit > 0:
        imageID_chunks = array_split(imageIDs, batchSizeLimit)
    else:
        imageID_chunks = [imageIDs]

    # process in batches
    for idx, imageID_batch in enumerate(imageID_chunks):
        chunkStr = f'{idx+1}/{len(imageID_chunks)}'
        print(f'Chunk {chunkStr}')

        update_state = __get_message_fun(project, len(imageIDs), idx*batchSizeLimit, epoch, numEpochs)

        # load remaining data (image filenames, class definitions)
        update_state(state='PREPARING', message=f'[Epoch {epoch}] loading metadata (chunk {chunkStr})')
        try:
            data = __load_metadata(project, dbConnector, imageID_batch, False, modelOriginID)
        except Exception as e:
            print(e)
            raise Exception(f'[Epoch {epoch}] error during metadata loading (chunk {chunkStr})')

        # call inference function
        update_state(state='PREPARING', message=f'[Epoch {epoch}] starting inference (chunk {chunkStr})')
        try:
            result = modelInstance.inference(stateDict=stateDict, data=data, updateStateFun=update_state)
        except Exception as e:
            print(e)
            raise Exception(f'[Epoch {epoch}] error during inference (chunk {chunkStr}; reason: {str(e)})')

        # call ranking function (AL criterion)
        if rankerInstance is not None and hasattr(rankerInstance, 'rank'):
            update_state(state='PREPARING', message=f'[Epoch {epoch}] calculating priorities (chunk {chunkStr})')
            try:
                result = rankerInstance.rank(data=result, updateStateFun=update_state, **{'stateDict':stateDict})
            except Exception as e:
                print(e)
                raise Exception(f'[Epoch {epoch}] error during ranking (chunk {chunkStr}, reason: {str(e)})')

        # parse result
        try:
            update_state(state='FINALIZING', message=f'[Epoch {epoch}] saving predictions (chunk {chunkStr})')
            fieldNames = list(getattr(FieldNames_prediction, predType).value)
            fieldNames.append('image')      # image ID
            fieldNames.append('cnnstate')   # model state ID
            values_pred = []
            values_img = []     # mostly for feature vectors
            for imgID in result.keys():
                for prediction in result[imgID]['predictions']:

                    # if segmentation mask: encode
                    if predType == 'segmentationMasks':
                        segMask = np.array(result[imgID]['predictions'][0]['label']).astype(np.uint8)
                        height, width = segMask.shape
                        segMask = base64.b64encode(segMask.ravel()).decode('utf-8')
                        segMaskDimensions = {
                            'width': width,
                            'height': height
                        }

                    nextResultValues = []
                    # we expect a dict of values, so we can use the fieldNames directly
                    for fn in fieldNames:
                        if fn == 'image':
                            nextResultValues.append(imgID)
                            # ids_img.append(imgID)
                        elif fn == 'cnnstate':
                            nextResultValues.append(stateDictID)
                        elif fn == 'segmentationmask':
                            nextResultValues.append(segMask)
                        elif fn == 'width' or fn == 'height':
                            if predType == 'segmentationMasks':
                                nextResultValues.append(segMaskDimensions[fn])
                            elif fn in prediction:
                                nextResultValues.append(prediction[fn])
                            else:
                                nextResultValues.append(None)
                        elif fn == 'priority':
                            value = None
                            if fn in prediction and prediction[fn] is None and 'confidence' in prediction:
                                # ranker somehow didn't assign value; use confidence by default
                                value = prediction['confidence']
                            elif fn in prediction:
                                value = prediction[fn]
                            else:
                                #TODO: provide replacement for priority, e.g. in case of segmentation masks
                                value = None
                            
                            if isinstance(value, list) or isinstance(value, np.ndarray):
                                # segmentation masks, etc.: array of values provided; take average instead
                                try:
                                    value = np.nanmean(np.array(value))
                                except:
                                    value = None
                            nextResultValues.append(value)
                        elif fn == 'confidence':
                            value = None
                            if fn in prediction:
                                value = prediction[fn]
                                if isinstance(value, list) or isinstance(value, np.ndarray):
                                    # segmentation masks, etc.: array of values provided; take average instead
                                    try:
                                        value = np.nanmean(np.array(value))
                                    except:
                                        value = None
                            nextResultValues.append(value)
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
            raise Exception(f'[Epoch {epoch}] error during result parsing (chunk {chunkStr}, reason: {str(e)})')


        # commit to database
        try:
            if len(values_pred):
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
            raise Exception(f'[Epoch {epoch}] error during data committing (chunk {chunkStr}, reason: {str(e)})')
    
    update_state(state=states.SUCCESS, message='predicted on {} images'.format(len(imageIDs)))

    print(f'[{project}] Epoch {epoch}: Inference completed successfully.')
    return