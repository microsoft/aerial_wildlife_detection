# Launches the AIController and one AIWorker on the current machine.
# Includes launching of the Celery server.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger


# Settings filepath
export AIDE_CONFIG_PATH=config/settings.ini

#TODO
export AIDE_CONFIG_PATH=settings_windowCropping.ini


# Celery
# celery -A modules.AIController.backend.celery_interface worker &


# HTTP
python runserver.py --settings_filepath=settings_windowCropping.ini --instance=FileServer