'''
    Trainer for the U-Net by Ronneberger et al. (2015),
    implemented in PyTorch.

    2020 Benjamin Kellenberger
'''

import json
from .._segmentation import SegmentationModel
from ...functional.segmentationMasks.unet import UNet as Model
from ...functional.datasets.segmentationDataset import SegmentationDataset
from ._default_options import DEFAULT_OPTIONS


class UNet(SegmentationModel):

    model_class = Model

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(UNet, self).__init__(project, config, dbConnector, fileServer,
            options, UNet.getDefaultOptions())
        self.model_class = Model
        self.dataset_class = SegmentationDataset


    @staticmethod
    def getDefaultOptions():
        try:
            # try to load defaults from JSON file first
            options = json.load(open('config/ai/model/pytorch/segmentationMasks/unet.json', 'r'))
        except Exception:
            # error; fall back to built-in defaults
            options = DEFAULT_OPTIONS
        return options