'''
    Wrapper loading the WSODPointModel for point predictions.

    2019 Benjamin Kellenberger
'''

from ._points import PointModel
from ..functional._wsodPoints.model import WSODPointModel as Model
from ..functional.datasets.pointsDataset import PointsDataset


class WSODPointModel(PointModel):

    model_class = Model

    def __init__(self, config, dbConnector, fileServer, options):
        super(WSODPointModel, self).__init__(config, dbConnector, fileServer, options)
        self.model_class = Model
        self.dataset_class = PointsDataset