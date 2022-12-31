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
        help='Author name of the model. ' + \
            'If omitted, the current system\'s user account name will be used.')
    parser.add_argument('--tags', nargs='?',
        help='Tags to add to Model Marketplace instance. ' + \
            'Can be a list of strings or a path to a text or JSON file.')
    parser.add_argument('--citation-info', type=str,
        help='Citation information for model; supports basic HTML tags. ' + \
            'Can be a string or a path to a text file.')
    parser.add_argument('--license', type=str,
        help='License information. Can be a string or a path to a text file.')
    if add_labelclasses:
        parser.add_argument('--labelclasses', nargs='?',
            help='Labelclasses. ' + \
                'Can be a list of names or a path to a text or JSON file defining classes.')

    return parser



def parse_argument(arg_name, arg, target_type=str, check_for_file_path=True):
    def _extract_list_from_dict(input_object):
        # need list but got JSON dict; try to find longest list in keys
        keys = [key for key in input_object.keys() if isinstance(input_object[key], list)]
        list_sizes = [len(input_object[key]) for key in keys]
        if len(list_sizes) == 0:
            raise Exception(f'Argument "{arg_name}" points to JSON file, but no list found within.')
        largest = np.argmax(list_sizes)
        return input_object[keys[largest.tolist()]]

    if isinstance(arg, str):
        if target_type is not str:
            raise Exception(f'Argument "{arg_name}" should be "{target_type}", got "{type(arg)}".')
        if check_for_file_path and os.path.exists(arg):
            # load file
            _, ext = os.path.splitext(arg)
            if ext.lower() == '.json':
                with open(arg, 'r', encoding='utf-8') as f_json:
                    arg = json.load(f_json)
            else:
                try:
                    with open(arg, 'r', encoding='utf-8') as f_plain:
                        arg = f_plain.readlines()
                except Exception as exc:
                    raise Exception(f'Argument "{arg_name}" set to file "{arg}", ' +\
                        'but file could not be loaded.') from exc

        if target_type is str:
            if isinstance(arg, dict):
                arg = json.dumps(arg, indent=2)
            elif isinstance(arg, list):
                arg = '\n'.join(arg)
            return arg

        if target_type is list:
            # extract list
            if isinstance(arg, dict):
                return _extract_list_from_dict(arg)
            if isinstance(arg, list):
                return arg

    elif isinstance(arg, list):
        if target_type is list:
            return arg
        if target_type is str:
            return '\n'.join(arg)



def assemble_marketplace_metadata(model_library,
                                    labelclasses: list,
                                    model_name: str,
                                    description: str=None,
                                    author: str=None,
                                    tags: list=[],
                                    citation_info: str=None,
                                    license_info: str=None) -> dict:
    assert model_library in PREDICTION_MODELS, \
        f'Model library "{model_library}" could not be found. ' + \
            f'Must be one of: {" ".join(list(PREDICTION_MODELS.keys()))}'
    assert len(model_name), 'Model name must not be empty'
    assert len(labelclasses), 'Label class list must not be empty'
    if not isinstance(author, str):
        author = os.getlogin()
        print(f'Warning: no author name provided; replaced with "{author}".')

    model_library_meta = PREDICTION_MODELS[model_library]

    meta = MARKETPLACE_METADATA.copy()
    meta['name'] = model_name
    meta['description'] = description
    meta['ai_model_library'] = model_library
    meta['annotation_type'] = model_library_meta['annotationType']
    meta['prediction_type'] = model_library_meta['predictionType']
    meta['author'] = author
    meta['tags'] = tags if isinstance(tags, list) else []
    meta['citation_info'] = citation_info
    meta['license'] = license_info
    meta['time_created'] = datetime.now().timestamp()

    return meta
