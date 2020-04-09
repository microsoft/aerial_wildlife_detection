# Launches or terminates AIDE with all the correct modules,
# including Celery, if required.
# The information on which modules and services to run come
# from the following two environment variables:
# * AIDE_CONFIG_PATH
# * AIDE_MODULES
#
# Depending on the argume
# 
# 2020 Benjamin Kellenberger

function start {

    IFS=',' read -ra ADDR <<< "$AIDE_MODULES"

    # Celery
    numCeleryModules=0;
    for i in "${ADDR[@]}"; do
        module="$(echo "$i" | tr '[:upper:]' '[:lower:]')";
        if [ "$module" == "aiworker" ] || [ "$module" == "fileserver" ]; then
            ((numCeleryModules++));
        fi
    done
    if [ $numCeleryModules -gt 0 ]; then
        celery -A celery_worker worker -Q aide_broadcast,$AIDE_MODULES &
    else
        echo "Machine does not need a Celery consumer to be launched; skipping..."
    fi

    # AIDE
    numHTTPmodules=0;
    for i in "${ADDR[@]}"; do
        module="$(echo "$i" | tr '[:upper:]' '[:lower:]')";
        if [ "$module" != "aiworker" ]; then
            ((numHTTPmodules++));
        fi
    done

    if [ $numHTTPmodules -gt 0 ]; then
        # get host and port from configuration file
        host=$(python util/configDef.py --section=Server --parameter=host)
        port=$(python util/configDef.py --section=Server --parameter=port)
        numWorkers=$(python util/configDef.py --section=Server --parameter=numWorkers --fallback=6)
        gunicorn application:app --bind=$host:$port --workers=$numWorkers
    else
        echo "Machine only runs as an AIWorker; skipping set up of HTTP web server..."
    fi
}

function stop {
    celery -A celery_worker control shutdown;
    pkill gunicorn;
}

function restart {
    stop;
    start;
}

mode=$1;

if [ "$mode" == "start" ]; then
    start;
elif [ "$mode" == "restart" ]; then
    restart;
elif [ "$mode" == "stop" ]; then
    stop;
fi