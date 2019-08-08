'''
    Little helper that commits a PyTorch model state from a local path to the DB.

    2019 Benjamin Kellenberger
'''

import os
import io
import argparse
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load model state from disk and commit to DB.')
    parser.add_argument('--settings_filepath', type=str, default='settings_windowCropping.ini', const=1, nargs='?',
                    help='Directory of the settings.ini file used for this machine (default: "config/settings.ini").')
    parser.add_argument('--modelPath', type=str, default='/datadrive/projects/ai4edevelopment/BeniKellenberger2019/cnn_states/ste/retinanet_resnet18/laterEpochs/199.pth', const=1, nargs='?',
                    help='Directory (absolute path) on this machine of the model state file to be considered.')
    args = parser.parse_args()
    

    # setup
    print('Setup...')
    if not 'AIDE_CONFIG_PATH' in os.environ:
        os.environ['AIDE_CONFIG_PATH'] = str(args.settings_filepath)

    from util.configDef import Config
    from modules.Database.app import Database
    config = Config()
    dbConn = Database(config)


    # load model class function
    print('Load and verify state dict...')
    from util.helpers import get_class_executable, current_time
    modelClass = get_class_executable('ai.models.pytorch.functional._retinanet.model.RetinaNet')   #config.getProperty('AIController', 'model_lib_path'))     #TODO

    # load state dict
    stateDict = torch.load(open(args.modelPath, 'rb'))

    # append label class map definition (if not present)
    if not 'labelclassMap' in stateDict:
        # query label class definitions
        labelClasses = dbConn.execute('SELECT * FROM {schema}.labelclass;'.format(schema=config.getProperty('Database', 'schema')), None, 'all')
        labelclassMap = {}
        for idx, lc in enumerate(labelClasses):
            labelclassMap[lc['id']] = idx
        stateDict['labelclassMap'] = labelclassMap

    # verify model state
    model = modelClass.loadFromStateDict(stateDict)
    bio = io.BytesIO()
    torch.save(model.getStateDict(), bio)
    stateDict = bio.getvalue()

    # commit to DB
    print('Commit to DB...')
    sql = '''
        INSERT INTO {schema}.cnnstate (timeCreated, stateDict, partial)
        VALUES ( %s, %s, FALSE);
    '''.format(schema=config.getProperty('Database', 'schema'))
    now = current_time()
    dbConn.execute(sql, (now, stateDict, ), None)