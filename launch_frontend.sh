# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019-20 Benjamin Kellenberger


# get details from configuration file
host=$(python util/configDef.py --section=Server --parameter=host)
port=$(python util/configDef.py --section=Server --parameter=port)
numWorkers=$(python util/configDef.py --section=Server --parameter=numWorkers --fallback=6)


# HTTP server
gunicorn application:app --bind=$host:$port --workers=$numWorkers