'''
    Wrapper/entry point for WSGI servers like Gunicorn.
    Can launch multiple modules at once, similar to the "runserver" file,
    but requires environment variables to be set to do so.

    2019 Benjamin Kellenberger
'''


#TODO: manually set env. variables for development stage
import os
os.environ['AIDE_CONFIG_PATH'] = 'settings_wcsaerialblobs.ini'
os.environ['AIDE_MODULES'] = 'LabelUI,AIController'


''' import resources and initialize app '''
from bottle import Bottle
from util.configDef import Config
from modules import REGISTERED_MODULES


def _verify_unique(instances, moduleClass):
        '''
            Compares the newly requested module, address and port against
            already launched modules on this instance.
            Raises an Exception if another module from the same type has already been launched on this instance
        '''
        for i in instances:
            if moduleClass.__class__.__name__ == i.__class__.__name__:
                raise Exception('Module {} already launched on this server.'.format(moduleClass.__class__.__name__))


config = Config()

# set host and port from configuration
os.environ['GUNICORN_CMD_ARGS'] = '--bind={host}:{port} --workers=5'.format(
    host=config.getProperty('Server', 'host'),
    port=config.getProperty('Server', 'port')
)


# prepare bottle
app = Bottle()

# parse requested instances
instance_args = os.environ['AIDE_MODULES'].split(',')
instances = []

# create user handler
userHandler = REGISTERED_MODULES['UserHandler'](config, app)

for i in instance_args:

    moduleName = i.strip()
    if moduleName == 'UserHandler':
        continue

    moduleClass = REGISTERED_MODULES[moduleName]
    
    # verify
    _verify_unique(instances, moduleClass)

    # create instance
    instance = moduleClass(config, app)
    instances.append(instance)

    # add authentication functionality
    if hasattr(instance, 'addLoginCheckFun'):
        instance.addLoginCheckFun(userHandler.checkAuthenticated)


if __name__ == '__main__':
    app.run()