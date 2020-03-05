# Launches a Celery consumer on the current machine.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019-20 Benjamin Kellenberger

# Celery
celery -A celery_worker worker --hostname multibranch@%h
# celery -A modules.AIController.backend.celery_interface worker --hostname multibranch@%h