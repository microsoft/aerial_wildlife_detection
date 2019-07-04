'''
    Register modules here. Module-specific parameters in the config .ini file
    can be added under a section with the same name as the module.
'''

from .LabelUI.app import LabelUI
from .AIController.app import AIController
from .AIWorker.app import AIWorker
from .Database.app import Database
from .FileServer.app import FileServer
from .UserHandling.app import UserHandler


REGISTERED_MODULES = {
    'LabelUI': LabelUI,
    'AIController': AIController,
    'AIWorker': AIWorker,
    'Database': Database,
    'FileServer': FileServer,
    'UserHandler': UserHandler
}