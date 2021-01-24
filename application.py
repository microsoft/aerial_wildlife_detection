'''
    Wrapper/entry point for WSGI servers like Gunicorn.
    Can launch multiple modules at once,
    but requires environment variables to be set to do so.

    2019-20 Benjamin Kellenberger
'''

from setup.assemble_server import assemble_server

# we are launching Bottle from a web server like Gunicorn, so we skip the pre-flight check and are not verbose
app = assemble_server(verbose_start=False,
                    check_v1_config=False,
                    migrate_database=False,
                    force_migrate=False,
                    passive_mode=False)