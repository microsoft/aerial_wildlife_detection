# Launching AIde

The instructions below manually launch AIde using the [Gunicorn server](https://gunicorn.org/).
If you wish to deploy AIde properly, you might want to set up Gunicorn as a service and wrap it in a web server. To do so, see [here](deployment.md).



## Environment variables
Machines running an AIde service need to have the `AIDE_CONFIG_PATH` environment variable set:

* `AIDE_CONFIG_PATH`: location (relative or absolute path) to the [configuration *.ini file](configure_settings.md) on the current machine.


Setting this environment variables can be done temporarily (example):
```bash
    export AIDE_CONFIG_PATH=config/settings.ini
```

Or permanently (requires re-login):
```bash
    echo "export AIDE_CONFIG_PATH=config/settings.ini" | tee ~/.profile
```


## Frontend
The front-end modules of AIde (_LabelUI_, _AIController_, _FileServer_) can be run by launching Gunicorn with the correct Bottle application, accessible through the file `application.py` (`application:app`).

To launch just the frontend:
```bash
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini
    export AIDE_MODULES=LabelUI

    # get host and port from configuration file
    host=$(python util/configDef.py --section=Server --parameter=host)
    port=$(python util/configDef.py --section=Server --parameter=port)


    # launch gunicorn. You may want to configure the server with respective arguments.
    # See here: http://docs.gunicorn.org/en/latest/configure.html
    gunicorn application:app -b=$host:$port
```

Modules in the `AIDE_MODULES` variable can be chained with commas. For example, to launch the _LabelUI_, _AIController_ and _FileServer_ on the same machine, set the environment variable as follows instead:
```bash
    export AIDE_MODULES=LabelUI,AIController,FileServer
```


If, for some reason, you wish to launch AIde using Bottle directly (defaults to Python's built-in WSGI server), you can do so by launching `application.py` directly after setting environment variables `AIDE_CONFIG_PATH` and `AIDE_MODULES` accordingly:
```bash
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini
    export AIDE_MODULES=LabelUI

    # launch Bottle directly
    python application.py
```

Note that this is discouraged, though. 



## AIWorker
_AIWorker_ modules need to be launched using Celery:
```bash
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini

    # launch Celery worker
    celery -A modules.AIController.backend.celery_interface worker
```
