'''
    Wrapper/entry point for WSGI servers like Gunicorn.
    Can launch multiple modules at once,
    but requires environment variables to be set to do so.

    2019-20 Benjamin Kellenberger
'''


''' import resources and initialize app '''
import os
import sys
from bottle import Bottle
from util.helpers import LogDecorator
from util.configDef import Config
from setup.migrate_aide import migrate_aide
from modules import REGISTERED_MODULES, Database
from constants.version import AIDE_VERSION

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
config = Config(verbose_start=True)

# connect to database
dbConnector = Database(config, verbose_start=True)

# check if config file points to unmigrated v1 project
print('Checking database...'.ljust(10), end='')
hasAdminTable = dbConnector.execute('''
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'aide_admin'
            AND table_name = 'project'
        );
    ''', None, 1)
if not hasAdminTable[0]['exists']:
    # not (yet) migrated, raise Exception with instructions to ensure compatibility
    LogDecorator.print_status('fail')
    print(f'''
The current installation of AIDE:
    database host: {config.getProperty('Database', 'host')}
    database name: {config.getProperty('Database', 'name')}
    schema:        {config.getProperty('Database', 'schema', str, '(not specified)')}

points to an installation of the legacy AIDE v1.
If you wish to continue using AIDE v2, you have to upgrade the project accordingly.
For instructions to do so, see here:
    https://github.com/microsoft/aerial_wildlife_detection/blob/multiProject/doc/upgrade_from_v1.md
    ''')
    sys.exit(1)

else:
    LogDecorator.print_status('ok')

# check if project has been migrated
print('Checking projects...'.ljust(10), end='')
dbSchema = config.getProperty('Database', 'schema', str, None)
if dbSchema is not None:
    isMigrated = dbConnector.execute('''
            SELECT COUNT(*) AS cnt
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (dbSchema,), 1)
    if isMigrated is not None and len(isMigrated) and isMigrated[0]['cnt'] == 0:
        LogDecorator.print_status('warn')
        print(f'''
WARNING: the selected configuration .ini file
("{os.environ['AIDE_CONFIG_PATH']}")
points to a project that has not yet been migrated to AIDE v2.
Details:
    database host: {config.getProperty('Database', 'host')}
    database name: {config.getProperty('Database', 'name')}
    schema:        {dbSchema}

If you wish to continue using AIDE v2 for this project, you have to upgrade it
to v2 accordingly.
For instructions to do so, see here:
    https://github.com/microsoft/aerial_wildlife_detection/blob/multiProject/doc/upgrade_from_v1.md
    ''')
    else:
        LogDecorator.print_status('ok')


# bring AIDE up-to-date
print('Updating database...'.ljust(10), end='')
warnings, errors = migrate_aide()
if len(warnings) or len(errors):

    if len(errors):
        LogDecorator.print_status('fail')
    else:
        LogDecorator.print_status('warn')

    print(f'Warnings and/or errors occurred while updating AIDE to the latest version ({AIDE_VERSION}):')
    print('\nWarnings:')
    for w in warnings:
        print(f'\t"{w}"')
    
    print('\nErrors:')
    for e in errors:
        print(f'\t"{e}"')
    
    if len(errors):
        sys.exit(2)

else:
    LogDecorator.print_status('ok')


# prepare bottle
app = Bottle()

# parse requested instances
print('Launching AIDE modules...')
instance_args = os.environ['AIDE_MODULES'].split(',')
instances = {}

# create user handler
userHandler = REGISTERED_MODULES['UserHandler'](config, app)

# "singleton" Task Coordinator instance
taskCoordinator = REGISTERED_MODULES['TaskCoordinator'](config, app, verbose_start=True)
taskCoordinator.addLoginCheckFun(userHandler.checkAuthenticated)

for i in instance_args:

    moduleName = i.strip()
    if moduleName == 'UserHandler':
        continue

    moduleClass = REGISTERED_MODULES[moduleName]
    
    # verify
    _verify_unique(instances, moduleClass)

    # create instance
    instance = moduleClass(config, app, verbose_start=True)
    instances[moduleName] = instance

    # add authentication functionality
    if hasattr(instance, 'addLoginCheckFun'):
        instance.addLoginCheckFun(userHandler.checkAuthenticated)

    
    # launch project meta modules
    if moduleName == 'LabelUI':
        aideAdmin = REGISTERED_MODULES['AIDEAdmin'](config, app)
        aideAdmin.addLoginCheckFun(userHandler.checkAuthenticated)
        reception = REGISTERED_MODULES['Reception'](config, app)
        reception.addLoginCheckFun(userHandler.checkAuthenticated)
        configurator = REGISTERED_MODULES['ProjectConfigurator'](config, app)
        configurator.addLoginCheckFun(userHandler.checkAuthenticated)
        statistics = REGISTERED_MODULES['ProjectStatistics'](config, app)
        statistics.addLoginCheckFun(userHandler.checkAuthenticated)

    elif moduleName == 'FileServer':
        from modules.DataAdministration.backend import celery_interface as daa_int

    elif moduleName == 'AIController':
        from modules.AIController.backend import celery_interface as aic_int

        # launch model marketplace with AIController
        modelMarketplace = REGISTERED_MODULES['ModelMarketplace'](config, app, taskCoordinator)
        modelMarketplace.addLoginCheckFun(userHandler.checkAuthenticated)

    elif moduleName == 'AIWorker':
        from modules.AIWorker.backend import celery_interface as aiw_int


    # launch globally required modules
    dataAdmin = REGISTERED_MODULES['DataAdministrator'](config, app, taskCoordinator)
    dataAdmin.addLoginCheckFun(userHandler.checkAuthenticated)

    staticFiles = REGISTERED_MODULES['StaticFileServer'](config, app)
    staticFiles.addLoginCheckFun(userHandler.checkAuthenticated)
    


if __name__ == '__main__':

    # run using server selected by Bottle
    print('Launching server...')
    host = config.getProperty('Server', 'host')
    port = config.getProperty('Server', 'port')
    app.run(host=host, port=port)