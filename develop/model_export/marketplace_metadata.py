'''
    Adds basic metadata required for AIDE's Model Marketplace.

    2022 Benjamin Kellenberger
'''

import os
from datetime import datetime
import json
import numpy as np

from constants.version import AIDE_VERSION
from modules.ModelMarketplace.backend.marketplaceWorker import ModelMarketplaceWorker
from ai import PREDICTION_MODELS


MARKETPLACE_METADATA = {
    'aide_version': AIDE_VERSION,
    'aide_model_version': ModelMarketplaceWorker.MAX_AIDE_MODEL_VERSION,
    'time_created': datetime.now().timestamp(),

    'name': None,
    'description': '',
    'author': None,
    'tags': [],
    'labelclasses': None,
    'ai_model_library': None,
    'annotation_type': None,
    'prediction_type': None,
    'citation_info': None,
    'license': None
}



def add_meta_args(parser, add_labelclasses=True):
    parser.add_argument('--model-name', type=str,
                        help='Name of the model as it will appear in the AIDE Model Marketplace.')
    parser.add_argument('--description', type=str,
                        help='Model description. Can either be a text or a path to a text file.')
    parser.add_argument('--author', type=str,
                        help='Author name of the model. If omitted, the current system\'s user account name will be used.')
    parser.add_argument('--tags', nargs='?',
                        help='Tags to add to Model Marketplace instance. Can be a list of strings or a path to a text or JSON file.')
    parser.add_argument('--citation-info', type=str,
                        help='Citation information for model; supports basic HTML tags. Can be a string or a path to a text file.')
    parser.add_argument('--license', type=str,
                        help='License information. Can be a string or a path to a text file.')
    if add_labelclasses:
        parser.add_argument('--labelclasses', nargs='?',
                            help='Labelclasses. Can be a list of names or a path to a text or JSON file defining classes.')
    
    return parser



def parse_argument(arg_name, arg, target_type=str, check_for_file_path=True):
    def _extract_list_from_dict(input):
        # need list but got JSON dict; try to find longest list in keys
        keys = [key for key in input.keys() if isinstance(input[key], list)]
        listSizes = [len(input[key]) for key in keys]
        if not len(listSizes):
            raise Exception(f'Argument "{arg_name}" points to JSON file, but no list found within.')
        largest = np.argmax(listSizes)
        return input[keys[largest.tolist()]]

    if isinstance(arg, str):
        if target_type is not str:
            raise Exception(f'Argument "{arg_name}" should be "{target_type}", got "{type(arg)}".')
        if check_for_file_path and os.path.exists(arg):
            # load file
            _, ext = os.path.splitext(arg)
            if ext.lower() == '.json':
                arg = json.load(open(arg, 'r'))
            else:
                try:
                    with open(arg, 'r') as f:
                        arg = f.readlines()
                except Exception:
                    raise Exception(f'Argument "{arg_name}" set to file "{arg}", but file could not be loaded.')

        if target_type is str:
            if isinstance(arg, dict):
                arg = json.dumps(arg, indent=2)
            elif isinstance(arg, list):
                arg = '\n'.join(arg)
            return arg
        
        elif target_type is list:
            # extract list
            if isinstance(arg, dict):
                return _extract_list_from_dict(arg)
            elif isinstance(arg, list):
                return arg

    elif isinstance(arg, list):
        if target_type is list:
            return arg
        elif target_type is str:
            return '\n'.join(arg)



def assemble_marketplace_metadata(modelLibrary, labelclasses: list, modelName: str, description=None, author=None, tags=[], citation_info=None, license=None):
    
    assert modelLibrary in PREDICTION_MODELS, 'Model library "{}" could not be found. Must be one of: {}'.format(
        modelLibrary,
        ' '.join(list(PREDICTION_MODELS.keys()))
    )
    assert len(modelName), 'Model name must not be empty'
    assert len(labelclasses), 'Label class list must not be empty'
    if not isinstance(author, str):
        author = os.getlogin()
        print(f'Warning: no author name provided; replaced with "{author}".')

    modelLibraryMeta = PREDICTION_MODELS[modelLibrary]

    meta = MARKETPLACE_METADATA.copy()
    meta['name'] = modelName
    meta['description'] = description
    meta['ai_model_library'] = modelLibrary
    meta['annotation_type'] = modelLibraryMeta['annotationType']
    meta['prediction_type'] = modelLibraryMeta['predictionType']
    meta['author'] = author
    meta['tags'] = tags if isinstance(tags, list) else []
    meta['citation_info'] = citation_info
    meta['license'] = license
    meta['time_created'] = datetime.now().timestamp()

    return meta