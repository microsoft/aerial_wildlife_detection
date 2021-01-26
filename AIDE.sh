# Launches or terminates AIDE with all the correct modules,
# including Celery, if required.
# The information on which modules and services to run come
# from the following two environment variables:
# * AIDE_CONFIG_PATH
# * AIDE_MODULES
# 
# 2020-21 Benjamin Kellenberger

function start {

    IFS=',' read -ra ADDR <<< "$AIDE_MODULES"

    # Celery
    numCeleryModules=0;
    appendMarketplace='';
    for i in "${ADDR[@]}"; do
        module="$(echo "$i" | tr '[:upper:]' '[:lower:]')";
        if [ "$module" == "labelui" ]; then
            ((numCeleryModules++));
            appendMarketplace='ModelMarketplace';
        elif [ "$module" == "aiworker" ] || [ "$module" == "fileserver" ]; then
            ((numCeleryModules++));
        fi
    done
    if [ $numCeleryModules -gt 0 ]; then
        celery -A celery_worker worker -Q aide_broadcast,$AIDE_MODULES,$appendMarketplace &
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
        # perform verbose pre-flight checks
        python setup/assemble_server.py

        if [ $? -eq 0 ]; then
            # pre-flight checks succeeded; get host and port from configuration file
            host=$(python util/configDef.py --section=Server --parameter=host)
            port=$(python util/configDef.py --section=Server --parameter=port)
            numWorkers=$(python util/configDef.py --section=Server --parameter=numWorkers --type=int --fallback=6)

            debug="$(echo "$2" | tr '[:upper:]' '[:lower:]')";
            if [ "$debug" == "debug" ]; then
                debug="--log-level debug";
            else
                debug="";
            fi
            gunicorn application:app --bind=$host:$port --workers=$numWorkers $debug
        else
            echo -e "\033[0;31mPre-flight checks failed; aborting launch of AIDE.\033[0m"
        fi
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
else
    echo "Usage: AIDE.sh {start|restart|stop} [debug]"
fi