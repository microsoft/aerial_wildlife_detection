'''
    Register modules here. Module-specific parameters in the config .ini file
    can be added under a section with the same name as the module.
'''

from .LabelUI.app import LabelUI
from .Database.app import Database
from .FileServer.app import FileServer
from .UserHandling.app import UserHandler
from .Reception.app import Reception
from .ProjectAdministration.app import ProjectConfigurator
from .ProjectStatistics.app import ProjectStatistics


REGISTERED_MODULES = {
    'LabelUI': LabelUI,
    # 'AIController': AIController,
    # 'AIWorker': AIWorker,
    'Database': Database,
    'FileServer': FileServer,
    'UserHandler': UserHandler,
    'Reception': Reception,
    'ProjectConfigurator': ProjectConfigurator,
    'ProjectStatistics': ProjectStatistics
}


#TODO: dirty hack...
try:
    from .AIController.app import AIController
    from .AIWorker.app import AIWorker
    REGISTERED_MODULES['AIController'] = AIController
    REGISTERED_MODULES['AIWorker'] = AIWorker
except:
    pass