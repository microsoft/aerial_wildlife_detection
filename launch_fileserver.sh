# Convenience function to run a FileServer instance.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# NEVER, EVER USE THIS FOR DEPLOYMENT!
# Instead, it is strongly recommended to use a proper file server like nginx.
#
# 2019 Benjamin Kellenberger


# # Settings filepath
# export AIDE_CONFIG_PATH=config/settings.ini
# #TODO
# export AIDE_CONFIG_PATH=settings_windowCropping.ini
# export AIDE_CONFIG_PATH=settings_wcsaerialblobs.ini

# modules to run
export AIDE_MODULES=FileServer

# HTTP server
python runserver.py --settings_filepath=$AIDE_CONFIG_PATH --instance=FileServer
#gunicorn application:app -b=0.0.0.0:8086