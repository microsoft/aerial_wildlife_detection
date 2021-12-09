'''
    Register modules here. Module-specific parameters in the config .ini file
    can be added under a section with the same name as the module.

    2019-2021 Benjamin Kellenberger
'''

# set up Celery configuration
import celery_worker

from .LabelUI.app import LabelUI
from .Database.app import Database
from .FileServer.app import FileServer
from .UserHandling.app import UserHandler
from .Reception.app import Reception
from .ProjectAdministration.app import ProjectConfigurator
from .ProjectStatistics.app import ProjectStatistics
from .DataAdministration.app import DataAdministrator
from .StaticFiles.app import StaticFileServer
from .AIDEAdmin.app import AIDEAdmin
from .ModelMarketplace.app import ModelMarketplace
from .TaskCoordinator.app import TaskCoordinator
from .ImageQuerying.app import ImageQuerier


#TODO
from .AIController.app import AIController
from .AIWorker.app import AIWorker


REGISTERED_MODULES = {
    'LabelUI': LabelUI,
    'AIController': AIController,
    'AIWorker': AIWorker,
    'Database': Database,
    'FileServer': FileServer,
    'UserHandler': UserHandler,
    'Reception': Reception,
    'ProjectConfigurator': ProjectConfigurator,
    'ProjectStatistics': ProjectStatistics,
    'DataAdministrator': DataAdministrator,
    'StaticFileServer': StaticFileServer,
    'AIDEAdmin': AIDEAdmin,
    'ModelMarketplace': ModelMarketplace,
    'TaskCoordinator': TaskCoordinator,
    'ImageQuerier': ImageQuerier
}