'''
    TODO: temporary script for debugging models.
    To be deleted upon completion.
'''

from ai.models.pytorch.points import WSODPointModel


if __name__ == '__main__':
    import os

    os.environ['AIDE_CONFIG_PATH'] = 'settings_objectCentered.ini'
    from util.configDef import Config
    from modules.Database.app import Database
    from modules.AIWorker.backend.worker.fileserver import FileServer
    config = Config()
    dbConnector = Database(config)
    fileServer = FileServer(config)

    rn = WSODPointModel(config, dbConnector, fileServer, None)


    # do inference on unlabeled
    def __load_model_state(config, dbConnector):
        # load model state from database
        sql = '''
            SELECT query.statedict FROM (
                SELECT statedict, timecreated
                FROM {schema}.cnnstate
                ORDER BY timecreated ASC NULLS LAST
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
    stateDict = __load_model_state(config, dbConnector)


    #TODO
    from constants.dbFieldNames import FieldNames_annotation
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
            fieldNames = list(getattr(FieldNames_annotation, config.getProperty('Project', 'annotationType')).value)
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

    sql = '''SELECT image FROM {schema}.image_user WHERE viewcount > 0 LIMIT 4096'''.format(schema=config.getProperty('Database', 'schema'))
    imageIDs = dbConnector.execute(sql, None, 128)
    imageIDs = [i['image'] for i in imageIDs]

    data = __load_metadata(config, dbConnector, imageIDs, True)

    stateDict = rn.train(stateDict, data)

    print('debug')

    rn.inference(stateDict, data)