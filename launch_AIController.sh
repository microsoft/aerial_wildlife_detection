# Launches a single-threaded AIController instance on this machine.
# Single-threading is important for the AIController in order to be
# able to always get the correct messages for all the clients
# polling the worker status. In a multi-threaded setting clients
# accessing one of the threaded instances that did not submit the
# job, all they would see is a message "job at worker" that would
# never terminate.
#
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019 Benjamin Kellenberger


# Settings filepath
export AIDE_CONFIG_PATH=config/settings.ini

#TODO
export AIDE_CONFIG_PATH=settings_windowCropping.ini
export AIDE_CONFIG_PATH=settings_wcsaerialblobs.ini


# get host and port from configuration file
host=$(python util/configDef.py --section=Server --parameter=host)
port=$(python util/configDef.py --section=Server --parameter=port)


# Gunicorn
gunicorn application:app -b=$host:$port -w 1