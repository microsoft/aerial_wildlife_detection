# Launches every module of the interface (LabelUI, AIController,
# AIWorker, FileServer, UserHandling), as well as the Celery server
# on the current machine.
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


# Celery
celery -A modules.AIController.backend.celery_interface worker &


# HTTP server
python runserver.py --settings_filepath=$AIDE_CONFIG_PATH --instance=UserHandler,FileServer,LabelUI,AIController