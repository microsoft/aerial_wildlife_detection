# Launches an AIWorker (Celery) on the current machine.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger
export AIDE_MODULES=LabelUI,AIController,FileServer
export AIDE_CONFIG_PATH=serengeti/serengeti-traps.ini

# Celery
celery -A modules.AIController.backend.celery_interface worker -Q aide
