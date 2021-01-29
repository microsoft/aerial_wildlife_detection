'''
    2021 Benjamin Kellenberger
'''

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.labels.torchvisionClassifier import GeneralizedTorchvisionClassifier, DEFAULT_OPTIONS


class ResNeXt50(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnext50_32x4d', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnext50_32x4d"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnext50.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnext50_32x4d_ImageNet.yaml'
        return opts


class ResNeXt101(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnext101_32x8d', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnext101_32x8d"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnext101.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnext101_32x8d_ImageNet.yaml'
        return opts