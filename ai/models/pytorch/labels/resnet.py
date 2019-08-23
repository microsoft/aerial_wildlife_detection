'''
    Wrapper loading a ResNet model for classification.

    2019 Benjamin Kellenberger
'''

from ._classification import ClassificationModel
from ..functional.classification.resnet import ResNet as Model
from ..functional.datasets.classificationDataset import LabelsDataset

class ResNet(ClassificationModel):

    model_class = Model

    def __init__(self, config, dbConnector, fileServer, options):
        super(ResNet, self).__init__(config, dbConnector, fileServer, options)
        self.model_class = Model
        self.dataset_class = LabelsDataset