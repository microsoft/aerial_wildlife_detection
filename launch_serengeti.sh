# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger

export AIDE_MODULES=LabelUI,AIController,FileServer
export AIDE_CONFIG_PATH=serengeti/serengeti-traps.ini

# get host and port from configuration file
host=$(python util/configDef.py --section=Server --parameter=host)
port=$(python util/configDef.py --section=Server --parameter=port)


# HTTP server
gunicorn application:app --bind=$host:$port --workers=1
