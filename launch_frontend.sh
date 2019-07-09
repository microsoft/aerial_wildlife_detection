# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger

# Terminate processes first
pkill celery;


# Settings filepath
export AIDE_CONFIG_PATH=config/settings.ini

#TODO
export AIDE_CONFIG_PATH=settings_windowCropping.ini


# HTTP server
python runserver.py --settings_filepath=$AIDE_CONFIG_PATH --instance=UserHandler,LabelUI,AIController