'''
    Creates and assembles the Bottle app with the individual
    AIDE modules.
    Can also be used as a dry run and to perform pre-flight
    checks (verbose start; database migration, etc.).

    2021 Benjamin Kellenberger
'''

import os
import sys
import platform
import argparse
import bottle
from bottle import Bottle
from util.helpers import LogDecorator
from util.configDef import Config
from setup.setupDB import add_update_superuser
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
            


def assemble_server(verbose_start=True, check_v1_config=True, migrate_database=True, force_migrate=False, passive_mode=False):

    # force verbosity if any of the pre-flight checks is enabled
    verbose_start = any((verbose_start, check_v1_config, migrate_database))

    instance_args = os.environ['AIDE_MODULES'].split(',')

    if verbose_start:
        configPath = os.environ['AIDE_CONFIG_PATH']
        aideModules = ', '.join(instance_args)

        print(f'''\033[96m
#################################       
                                        version {AIDE_VERSION}
   ###    #### ########  ########       
  ## ##    ##  ##     ## ##             {platform.platform()}
 ##   ##   ##  ##     ## ##             
##     ##  ##  ##     ## ######         [config]
#########  ##  ##     ## ##             .> {configPath}
##     ##  ##  ##     ## ##             
##     ## #### ########  ########       [modules]
                                        .> {aideModules}
#################################\033[0m
''')


    statusOffset = LogDecorator.get_ljust_offset()

    # load configuration
    config = Config(None, verbose_start)
    bottle.BaseRequest.MEMFILE_MAX = 1024**3    #TODO: make hyperparameter in config?

    # connect to database
    dbConnector = Database(config, verbose_start)

    if check_v1_config:
        # check if config file points to unmigrated v1 project
        print('Checking database...'.ljust(statusOffset), end='')
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

        # check if projects have been migrated
        print('Checking projects...'.ljust(statusOffset), end='')
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
        else:
            LogDecorator.print_status('ok')


    if migrate_database:
        # bring AIDE up-to-date
        print('Updating database...'.ljust(statusOffset), end='')
        warnings, errors = migrate_aide(force_migrate)
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

    # add/modify superuser credentials if needed
    try:
        result = add_update_superuser(config, dbConnector)
        if result['new_account']:
            print('New super user credentials found in configuration file and added to database:')
            print('\tName:   ' + result['details']['name'])
            print('\tE-mail: ' + result['details']['email'])
            print('\tPassword: ****')
        elif len(result['changes']):
            print('Super user account details changed for account name "". New credentials:')
            print('\tName:   ' + result['details']['name'])
            print('\tE-mail: ' + result['details']['email'] + (' (changed)' if result['changes'].get('adminEmail', False) else ''))
            print('\tPassword: ****' + (' (changed)' if result['changes'].get('adminPassword', False) else ''))
    except Exception as e:
        # no superuser credentials provided; ignore
        print(e)    #TODO
        pass

    # prepare bottle
    app = Bottle()

    # parse requested instances
    instances = {}

    # "singletons"
    dbConnector = REGISTERED_MODULES['Database'](config, verbose_start)
    userHandler = REGISTERED_MODULES['UserHandler'](config, app, dbConnector)
    taskCoordinator = REGISTERED_MODULES['TaskCoordinator'](config, app, dbConnector, verbose_start)
    taskCoordinator.addLoginCheckFun(userHandler.checkAuthenticated)

    for i in instance_args:

        moduleName = i.strip()
        if moduleName == 'UserHandler':
            continue

        moduleClass = REGISTERED_MODULES[moduleName]
        
        # verify
        _verify_unique(instances, moduleClass)

        # create instance
        if moduleName == 'AIController':
            instance = moduleClass(config, app, dbConnector, taskCoordinator, verbose_start, passive_mode)
        else:
            instance = moduleClass(config, app, dbConnector, verbose_start)
        instances[moduleName] = instance

        # add authentication functionality
        if hasattr(instance, 'addLoginCheckFun'):
            instance.addLoginCheckFun(userHandler.checkAuthenticated)

        
        # launch project meta modules
        if moduleName == 'LabelUI':
            aideAdmin = REGISTERED_MODULES['AIDEAdmin'](config, app, dbConnector, verbose_start)
            aideAdmin.addLoginCheckFun(userHandler.checkAuthenticated)
            reception = REGISTERED_MODULES['Reception'](config, app, dbConnector)
            reception.addLoginCheckFun(userHandler.checkAuthenticated)
            configurator = REGISTERED_MODULES['ProjectConfigurator'](config, app, dbConnector)
            configurator.addLoginCheckFun(userHandler.checkAuthenticated)
            statistics = REGISTERED_MODULES['ProjectStatistics'](config, app, dbConnector)
            statistics.addLoginCheckFun(userHandler.checkAuthenticated)

        elif moduleName == 'FileServer':
            from modules.DataAdministration.backend import celery_interface as daa_int

        elif moduleName == 'AIController':
            from modules.AIController.backend import celery_interface as aic_int

            # launch model marketplace with AIController
            modelMarketplace = REGISTERED_MODULES['ModelMarketplace'](config, app, dbConnector, taskCoordinator)
            modelMarketplace.addLoginCheckFun(userHandler.checkAuthenticated)

        elif moduleName == 'AIWorker':
            from modules.AIWorker.backend import celery_interface as aiw_int


        # launch globally required modules
        dataAdmin = REGISTERED_MODULES['DataAdministrator'](config, app, dbConnector, taskCoordinator)
        dataAdmin.addLoginCheckFun(userHandler.checkAuthenticated)

        staticFiles = REGISTERED_MODULES['StaticFileServer'](config, app, dbConnector)
        staticFiles.addLoginCheckFun(userHandler.checkAuthenticated)
    
    if verbose_start:
        print('\n')

    return app



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch AIDE server (single-threaded) or perform pre-flight checks.')
    parser.add_argument('--launch', type=int, default=0,
                        help='If set to 1, a single-threaded server (typically Python WSGI ref.) will be launched.')
    parser.add_argument('--check_v1', type=int, default=1,
                        help='Set to 1 to check database for unmigrated AIDE v1 setup.')
    parser.add_argument('--migrate_db', type=int, default=1,
                        help='Set to 1 to upgrade database with latest changes for AIDE setup.')
    parser.add_argument('--force_migrate', type=int, default=0,
                        help='If set to 1, database upgrade will be enforced even if AIDE versions already match.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Set to 1 to print launch information to console.')
    args = parser.parse_args()

    try:
        app = assemble_server(args.verbose, args.check_v1, args.migrate_db, args.force_migrate, not bool(args.launch))
    except Exception as e:
        print(e)
        sys.exit(1)

    if bool(args.launch):
        if args.verbose:
            print('Launching server...')
        config = Config(False)
        host = config.getProperty('Server', 'host')
        port = config.getProperty('Server', 'port')
        app.run(host=host, port=port)
    
    else:
        sys.exit(0)