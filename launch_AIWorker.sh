# Launches an AIWorker (Celery) on the current machine.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger


# # Settings filepath
# export AIDE_CONFIG_PATH=config/settings.ini

# #TODO
# export AIDE_CONFIG_PATH=settings_windowCropping.ini
# export AIDE_CONFIG_PATH=settings_wcsaerialblobs.ini


# Celery
celery -A modules.AIController.backend.celery_interface worker