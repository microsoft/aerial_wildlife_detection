'''
    Wrapper loading the WSODPointModel for point predictions.

    2019-20 Benjamin Kellenberger
'''

import json
from .._points import PointModel
from ...functional._wsodPoints.model import WSODPointModel as Model
from ...functional.datasets.pointsDataset import PointsDataset
from ._default_options import DEFAULT_OPTIONS


class WSODPointModel(PointModel):

    model_class = Model

    def __init__(self, project, config, dbConnector, fileServer, options):
        super(WSODPointModel, self).__init__(project, config, dbConnector, fileServer,
            options, WSODPointModel.getDefaultOptions())
        self.model_class = Model
        self.dataset_class = PointsDataset


    @staticmethod
    def getDefaultOptions():
        try:
            # try to load defaults from JSON file first
            options = json.load(open('config/ai/model/pytorch/points/wsodPointModel.json', 'r'))
        except:
            # error; fall back to built-in defaults
            options = DEFAULT_OPTIONS
        return options