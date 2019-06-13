'''
    Register modules here. Module-specific parameters in the config .ini file
    can be added under a section with the same name as the module.
'''
from .AITrainer.app import AITrainer
from .LabelUI.app import LabelUI
from .Database.app import Database
from .FileServer.app import FileServer
from .UserHandling.app import UserHandler


REGISTERED_MODULES = {
    'LabelUI': LabelUI,
    'AITrainer': AITrainer,
    'Database': Database,
    'FileServer': FileServer,
    'UserHandler': UserHandler
}