# Launches every module of the interface (LabelUI, AIController,
# AIWorker, FileServer, UserHandling), as well as the Celery server
# on the current machine.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger

# Terminate processes first
pkill celery;


# Celery
celery -A modules.AIController.backend.celery worker &


# HTTP
python runserver.py --settings_filepath=settings_windowCropping.ini --instance=UserHandler,FileServer,LabelUI,AIController,AIWorker