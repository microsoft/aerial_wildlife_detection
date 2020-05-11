'''
    Wrapper loading a ResNet model for classification.

    2019-20 Benjamin Kellenberger
'''

from .._classification import ClassificationModel
from ...functional.classification.resnet import ResNet as Model
from ...functional.datasets.classificationDataset import LabelsDataset
from ._default_options import DEFAULT_OPTIONS


class ResNet(ClassificationModel):

    model_class = Model

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(ResNet, self).__init__(project, config, dbConnector, fileServer,
            options, ResNet.getDefaultOptions())
        self.model_class = Model
        self.dataset_class = LabelsDataset


    @staticmethod
    def getDefaultOptions():
        try:
            # try to load defaults from JSON file first
            options = json.load(open('config/ai/model/pytorch/labels/resnet.json', 'r'))
        except:
            # error; fall back to built-in defaults
            options = DEFAULT_OPTIONS
        return options