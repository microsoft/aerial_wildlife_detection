# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger


# get host and port from configuration file
host=$(python util/configDef.py --section=Server --parameter=host)
port=$(python util/configDef.py --section=Server --parameter=port)


# HTTP server
gunicorn application:app --bind=$host:$port --workers=6