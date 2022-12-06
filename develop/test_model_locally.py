'''
    Debugging function that allows running an AI model locally
    (i.e., without the Celery routine). This still requires an
    instance of AIDE that is set up properly, as well as a pro-
    ject containing images (and optionally annotations) to ser-
    ve as a basis.

    Usage:
        python debug/test_model_local.py [project] [mode]
    
    With:
        - "project": shortname of the project to be used for
          testing
        - "mode": one of {'train', 'inference', 'TODO'}

    2020-22 Benjamin Kellenberger
'''

import os

# check if environment variables are set properly
assert 'AIDE_CONFIG_PATH' in os.environ and os.path.isfile(os.environ['AIDE_CONFIG_PATH']), \
    'ERROR: environment variable "AIDE_CONFIG_PATH" is not set.'
if not 'AIDE_MODULES' in os.environ:
    os.environ['AIDE_MODULES'] = 'LabelUI'
elif not 'labelui' in os.environ['AIDE_MODULES'].lower():
    # required for checking file server version
    os.environ['AIDE_MODULES'] += ',LabelUI'

import argparse
from constants.version import AIDE_VERSION
from util import helpers
from util.configDef import Config
from modules import REGISTERED_MODULES
from modules.Database.app import Database
from modules.AIDEAdmin.app import AdminMiddleware
from modules.AIController.backend.functional import AIControllerWorker
from modules.AIWorker.app import AIWorker
from modules.AIWorker.backend.fileserver import FileServer
from modules.AIWorker.backend.worker.functional import __load_model_state, __load_metadata


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='AIDE local model tester')
    parser.add_argument('--project', type=str, required=True,
                        help='Project shortname to draw sample data from.')
    parser.add_argument('--mode', type=str, required=True,
                        help='Evaluation mode (function to call). One of {"train", "inference"}.')
    parser.add_argument('--update-model', type=int, default=1,
                        help='Set to 1 (default) to perform a model update step prior to training or inference. ' + \
                            'This is required to e.g. adapt the model to new label classes in the project.')
    parser.add_argument('--modelLibrary', type=str, required=False,
                        help='Optional AI model library override. Provide a dot-separated Python import path here.')
    parser.add_argument('--modelSettings', type=str, required=False,
                        help='Optional AI model settings override (absolute or relative path to settings file, or else "none" to not use any predefined settings).')
    args = parser.parse_args()
    #TODO: continue

    assert args.mode.lower() in ('train', 'inference'), f'"{args.mode}" is not a known evaluation mode.'
    mode = args.mode.lower()

    # initialize required modules
    config = Config()
    dbConnector = Database(config)
    fileServer = FileServer(config).get_secure_instance(args.project)
    aiw = AIWorker(config, dbConnector, True)
    aicw = AIControllerWorker(config, None)

    # check if AIDE file server is reachable
    admin = AdminMiddleware(config, dbConnector)
    connDetails = admin.getServiceDetails(True, False)
    fsVersion = connDetails['FileServer']['aide_version']
    if not isinstance(fsVersion, str):
        # no file server running
        raise Exception('ERROR: AIDE file server is not running, but required for running models. Make sure to launch it prior to running this script.')
    elif fsVersion != AIDE_VERSION:
        print(f'WARNING: the AIDE version of File Server instance ({fsVersion}) differs from this one ({AIDE_VERSION}).')


    # get model trainer instance and settings
    queryStr = '''
        SELECT ai_model_library, ai_model_settings FROM aide_admin.project
        WHERE shortname = %s;
    '''
    result = dbConnector.execute(queryStr, (args.project,), 1)
    if result is None or not len(result):
        raise Exception(f'Project "{args.project}" could not be found in this installation of AIDE.')

    modelLibrary = result[0]['ai_model_library']
    modelSettings = result[0]['ai_model_settings']

    customSettingsSpecified = False
    if hasattr(args, 'modelSettings') and isinstance(args.modelSettings, str) and len(args.modelSettings):
        # settings override specified
        if args.modelSettings.lower() == 'none':
            modelSettings = None
            customSettingsSpecified = True
        elif not os.path.isfile(args.modelSettings):
            print(f'WARNING: model settings override provided, but file cannot be found ("{args.modelSettings}"). Falling back to project default ("{modelSettings}").')
        else:
            modelSettings = args.modelSettings
            customSettingsSpecified = True

    if hasattr(args, 'modelLibrary') and isinstance(args.modelLibrary, str) and len(args.modelLibrary):
        # library override specified; try to import it
        try:
            modelClass = helpers.get_class_executable(args.modelLibrary)
            if modelClass is None:
                raise
            modelLibrary = args.modelLibrary

            # re-check if current model settings are compatible; warn and set to None if not
            if modelLibrary != result[0]['ai_model_library'] and not customSettingsSpecified:
                # project model settings are not compatible with provided model
                print('WARNING: custom model library specified differs from the one currently set in project. Model settings will be set to None.')
                modelSettings = None

        except Exception as e:
            print(f'WARNING: model library override provided ("{args.modelLibrary}"), but could not be imported. Falling back to project default ("{modelLibrary}").')
        
    # initialize instance
    print(f'Using model library "{modelLibrary}".')
    modelTrainer = aiw._init_model_instance(args.project, modelLibrary, modelSettings)

    try:
        stateDict, _, modelOriginID, _ = __load_model_state(args.project, modelLibrary, dbConnector)    #TODO: load latest unless override is specified?
    except Exception:
        stateDict = None
        modelOriginID = None

    # helper functions
    def updateStateFun(state, message, done=None, total=None):
        print(message, end='')
        if done is not None and total is not None:
            print(f': {done}/{total}')
        else:
            print('')

    # launch task(s)
    if mode == 'train':
        data = aicw.get_training_images(
            project=args.project,
            maxNumImages=512)
        data = __load_metadata(args.project, dbConnector, data[0], True, modelOriginID)

        if bool(args.update_model):
            stateDict = modelTrainer.update_model(stateDict, data, updateStateFun)

        result = modelTrainer.train(stateDict, data, updateStateFun)
        if result is None:
            raise Exception('Training function must return an object (i.e., trained model state) to be stored in the database.')
        
    elif mode == 'inference':
        data = aicw.get_inference_images(
            project=args.project,
            maxNumImages=512)
        data = __load_metadata(args.project, dbConnector, data[0], False, modelOriginID)

        if bool(args.update_model):
            stateDict = modelTrainer.update_model(stateDict, data, updateStateFun)

        result = modelTrainer.inference(stateDict, data, updateStateFun)
        #TODO: check result for validity

if __name__ == '__main__':
    main()