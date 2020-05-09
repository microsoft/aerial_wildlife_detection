'''
    Trainer for the U-Net by Ronneberger et al. (2015),
    implemented in PyTorch.

    2020 Benjamin Kellenberger
'''

from ._segmentation import SegmentationModel
from ..functional.segmentationMasks.unet import UNet as Model
from ..functional.datasets.segmentationDataset import SegmentationDataset

class UNet(SegmentationModel):

    model_class = Model

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(UNet, self).__init__(project, config, dbConnector, fileServer, options)
        self.model_class = Model
        self.dataset_class = SegmentationDataset