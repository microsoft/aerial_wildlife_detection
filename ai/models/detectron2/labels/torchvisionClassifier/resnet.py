'''
    2021 Benjamin Kellenberger
'''

from ai.models.detectron2.genericDetectronModel import GenericDetectron2Model
from ai.models.detectron2.labels.torchvisionClassifier import GeneralizedTorchvisionClassifier, DEFAULT_OPTIONS


class ResNet18(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnet18', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnet18"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnet18.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnet_R_18_ImageNet.yaml'
        return opts


class ResNet34(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnet34', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnet34"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnet34.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnet_R_34_ImageNet.yaml'
        return opts


class ResNet50(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnet50', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnet50"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnet50.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnet_R_50_ImageNet.yaml'
        return opts


class ResNet101(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnet101', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnet101"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnet101.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnet_R_101_ImageNet.yaml'
        return opts


class ResNet152(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'resnet152', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "resnet152"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/resnet152.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/resnet_R_152_ImageNet.yaml'
        return opts

            
class WideResNet50(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'wide_resnet50_2', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "wide_resnet50_2"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/wideresnet50.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/wide_resnet50_2_ImageNet.yaml'
        return opts


class WideResNet101(GeneralizedTorchvisionClassifier):

    def __init__(self, project, config, dbConnector, fileServer, options):
        super().__init__(project, config, dbConnector, fileServer, options)
        assert self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR == 'wide_resnet101_2', \
            f'{self.detectron2cfg.MODEL.TVCLASSIFIER.FLAVOR} != "wide_resnet101_2"'

    @classmethod
    def getDefaultOptions(cls):
        opts = GenericDetectron2Model._load_default_options(
            'config/ai/model/detectron2/labels/wideresnet101.json',
            DEFAULT_OPTIONS
        )
        opts['defs']['model'] = 'labels/wide_resnet101_2_ImageNet.yaml'
        return opts