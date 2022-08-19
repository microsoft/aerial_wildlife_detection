'''
    Model export from DeepForest to AIDE Model Marketplace.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import io
from typing import OrderedDict
import json
import zipfile
import torch
from detectron2.config import get_cfg

from ai.models.detectron2.boundingBoxes.deepforest.deepforest import DeepForest
from develop.model_export import marketplace_metadata



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Exports a DeepForest model for import in AIDE\'s Model Marketplace')
    parser.add_argument('--checkpoint', type=str,
                        help='Path of the DeepForest model checkpoint to use.')
    parser.add_argument('--marketplace-file', type=str,
                        help='Path of the Model Marketplace metadata file.')
    parser.add_argument('--destination', type=str,
                        help='Path to save the Model Marketplace ZIP file to.')
    args = parser.parse_args()
    if args.marketplace_file is None or not os.path.exists(args.marketplace_file):
        # no default marketplace file; need to parse arguments via command line
        marketplace_metadata.add_meta_args(parser, add_labelclasses=True)
    args = parser.parse_args()

    # assertions
    assert os.path.exists(args.checkpoint), f'Model checkpoint "{args.checkpoint}" could not be found.'
    assert os.path.exists(args.marketplace_file), f'Model Marketplace metadata file "{args.marketplace_file}" could not be found.'

    # parse DeepForest model
    print(f'Loading and parsing checkpoint file "{args.checkpoint}"...')
    labelclasses = []
    checkpoint = torch.load(open(args.checkpoint, 'rb'), map_location='cpu')
    if isinstance(checkpoint, OrderedDict):
        stateDict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        stateDict = checkpoint['state_dict']

        # extract label classes if present
        if 'hyper_parameters' in checkpoint and 'label_dict' in checkpoint['hyper_parameters']:
            lcMap = checkpoint['hyper_parameters']['label_dict']
            lcMap = dict([[v,k] for k,v in lcMap.items()])      # PyTorch-lignting format: name : idx; we need to revert it
            lcKeys = list(lcMap.keys())
            lcKeys.sort()
            labelclasses = [lcMap[lcKey] for lcKey in lcKeys]
            print(f'Found {len(labelclasses)} label classes in state dict.')
    else:
        raise Exception(f'Could not load checkpoint "{args.checkpoint}"; no parseable parameters found.')

    # modify keys depending on whether state dict was created from deepforest or deepforest.model
    keys = list(stateDict.keys())
    if keys[0].startswith('model.'):
        # created using the main deepforest class; strip prefixes
        for key in keys:
            key_new = key.replace('model.', '')
            stateDict[key_new] = stateDict[key]
            del stateDict[key]
    
    # parse meta arguments
    print('Parsing and consolidating meta-arguments...')
    if labelclasses is None:
        labelclasses = marketplace_metadata.parse_argument('--labelclasses', args.labelClasses, target_type=list, check_for_file_path=True)
    description = marketplace_metadata.parse_argument('--description', args.description, target_type=str, check_for_file_path=True)
    tags = marketplace_metadata.parse_argument('--tags', args.tags, target_type=list, check_for_file_path=True)
    citation_info = marketplace_metadata.parse_argument('--citation-info', args.citation_info, target_type=str, check_for_file_path=True)
    license = marketplace_metadata.parse_argument('--license', args.license, target_type=str, check_for_file_path=True)

    # assemble metadata
    meta = marketplace_metadata.assemble_marketplace_metadata(
        'ai.models.detectron2.DeepForest',
        labelclasses,
        args.model_name,
        description,
        args.author,
        tags,
        citation_info,
        license
    )

    labelclassMap = dict(zip(labelclasses, list(range(len(labelclasses)))))

    # assemble AIDE options and Detectron2 cfg
    options = DeepForest.getDefaultOptions()

    cfg = get_cfg()
    DeepForest._add_deepforest_config(cfg)  #TODO: hyperparameters

    # assemble model state
    modelState = {
        'detectron2cfg': cfg,
        'labelclassMap': labelclassMap,
        'model': stateDict
    }

    # export to ZIP file
    print('Exporting to file...')
    with zipfile.ZipFile(args.destination, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('modelDefinition.json', json.dumps(meta))
        bio = io.BytesIO()
        torch.save(modelState, bio)
        zf.writestr('modelState.bin', bio.getvalue())
    
    print(f'\nSaved model state to file "{args.destination}".')