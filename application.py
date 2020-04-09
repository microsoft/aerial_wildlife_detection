'''
    Wrapper/entry point for WSGI servers like Gunicorn.
    Can launch multiple modules at once,
    but requires environment variables to be set to do so.

    2019-20 Benjamin Kellenberger
'''


''' import resources and initialize app '''
import os
from bottle import Bottle
from util.configDef import Config
from modules import REGISTERED_MODULES


def _verify_unique(instances, moduleClass):
        '''
            Compares the newly requested module, address and port against
            already launched modules on this instance.
            Raises an Exception if another module from the same type has already been launched on this instance
        '''
        for key in instances.keys():
            instance = instances[key]
            if moduleClass.__class__.__name__ == instance.__class__.__name__:
                raise Exception('Module {} already launched on this server.'.format(moduleClass.__class__.__name__))

# load configuration
config = Config()

# prepare bottle
app = Bottle()

# parse requested instances
instance_args = os.environ['AIDE_MODULES'].split(',')
instances = {}

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
    instances[moduleName] = instance

    # add authentication functionality
    if hasattr(instance, 'addLoginCheckFun'):
        instance.addLoginCheckFun(userHandler.checkAuthenticated)

    
    # launch project meta modules
    if moduleName == 'LabelUI':
        reception = REGISTERED_MODULES['Reception'](config, app)
        reception.addLoginCheckFun(userHandler.checkAuthenticated)
        configurator = REGISTERED_MODULES['ProjectConfigurator'](config, app)
        configurator.addLoginCheckFun(userHandler.checkAuthenticated)
        statistics = REGISTERED_MODULES['ProjectStatistics'](config, app)
        statistics.addLoginCheckFun(userHandler.checkAuthenticated)

    elif moduleName == 'FileServer':
        from modules.DataAdministration.backend import celery_interface as daa_int
        daa_int.aide_internal_notify({'task': 'add_projects'})

    elif moduleName == 'AIController':
        from modules.AIWorker.backend import celery_interface as aic_int        #TODO
        aic_int.aide_internal_notify({'task': 'add_projects'})


    # launch globally required modules
    dataAdmin = REGISTERED_MODULES['DataAdministrator'](config, app)
    dataAdmin.addLoginCheckFun(userHandler.checkAuthenticated)

    staticFiles = REGISTERED_MODULES['StaticFileServer'](config, app)
    staticFiles.addLoginCheckFun(userHandler.checkAuthenticated)
    


if __name__ == '__main__':

    # run using server selected by Bottle
    host = config.getProperty('Server', 'host')
    port = config.getProperty('Server', 'port')
    app.run(host=host, port=port)