'''
    2021 Benjamin Kellenberger
'''

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.labels.torchvisionClassifier import GeneralizedTorchvisionClassifier, DEFAULT_OPTIONS


class AlexNet(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'alexnet', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "alexnet"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/alexnet.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/alexnet_ImageNet.yaml'
        return opts