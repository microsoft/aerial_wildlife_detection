'''
    Wrapper/entry point for WSGI servers like Gunicorn.
    Can launch multiple modules at once,
    but requires environment variables to be set to do so.

    2019-20 Benjamin Kellenberger
'''


''' import resources and initialize app '''
import os
from bottle import Bottle
from setup.migrate_aide import migrate_aide
from util.configDef import Config
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
config = Config()

# check if config file points to unmigrated v1 project
dbConnector = Database(config)
hasAdminTable = dbConnector.execute('''
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'aide_admin'
            AND table_name = 'project'
        );
    ''', None, 1)
if not hasAdminTable[0]['exists']:
    # not (yet) migrated, raise Exception with instructions to ensure compatibility
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
    import sys
    sys.exit(1)

# check if project has been migrated
dbSchema = config.getProperty('Database', 'schema', str, None)
if dbSchema is not None:
    isMigrated = dbConnector.execute('''
            SELECT COUNT(*) AS cnt
            FROM aide_admin.project
            WHERE shortname = %s;
        ''', (dbSchema,), 1)
    if isMigrated is not None and len(isMigrated) and isMigrated[0]['cnt'] == 0:
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

# bring AIDE up-to-date
warnings, errors = migrate_aide()
if len(warnings) or len(errors):
    print(f'Warnings and/or errors occurred while updating AIDE to the latest version ({AIDE_VERSION}):')
    print('\nWarnings:')
    for w in warnings:
        print(f'\t"{w}"')
    
    print('\nErrors:')
    for e in errors:
        print(f'\t"{e}"')

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
        modelMarketplace = REGISTERED_MODULES['ModelMarketplace'](config, app)
        modelMarketplace.addLoginCheckFun(userHandler.checkAuthenticated)

    elif moduleName == 'AIWorker':
        from modules.AIWorker.backend import celery_interface as aiw_int


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