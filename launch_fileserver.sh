# Convenience function to run a FileServer instance.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# NEVER, EVER USE THIS FOR DEPLOYMENT!
# Instead, it is strongly recommended to use a proper file server like nginx.
#
# 2019-20 Benjamin Kellenberger

# modules to run
export AIDE_MODULES=FileServer

# get details from configuration file
host=$(python util/configDef.py --section=Server --parameter=host)
port=$(python util/configDef.py --section=Server --parameter=port)
numWorkers=$(python util/configDef.py --section=Server --parameter=numWorkers --fallback=6)


# HTTP server
gunicorn application:app --bind=$host:$port --workers=$numWorkers