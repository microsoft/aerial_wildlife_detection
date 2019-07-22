# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger


# Settings filepath
export AIDE_CONFIG_PATH=config/settings.ini
#TODO
export AIDE_CONFIG_PATH=settings_windowCropping.ini
export AIDE_CONFIG_PATH=settings_wcsaerialblobs.ini

# modules to run
export AIDE_MODULES=LabelUI,AIController

# HTTP server
#python runserver.py --settings_filepath=$AIDE_CONFIG_PATH --instance=UserHandler,LabelUI,AIController
gunicorn application:app -b=0.0.0.0:8086