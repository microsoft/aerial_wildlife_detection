#!/bin/sh

# Convenience function to run the frontend and AIController.
# Requires pwd to be the root of the project and the correct Python
# env to be loaded.
#
# 2019-22 Benjamin Kellenberger

python_exec=$1
if [ ${#python_exec} = 0 ]; then
    python_exec=$(command -v python)
fi

# pre-flight checks
$python_exec setup/assemble_server.py

if [ $? -eq 0 ]; then
    # pre-flight checks succeeded; get host and port from configuration file
    host=$(python3.8 util/configDef.py --section=Server --parameter=host)
    port=$(python3.8 util/configDef.py --section=Server --parameter=port)
    numWorkers=$(python3.8 util/configDef.py --section=Server --parameter=numWorkers --fallback=6)
    $python_exec -m gunicorn application:app --bind=$host:$port --workers=$numWorkers
else
    echo -e "\033[0;31mPre-flight checks failed; aborting launch of AIDE.\033[0m"
fi