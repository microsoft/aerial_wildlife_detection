# Launching AIDE

The instructions below manually launch AIDE using the [Gunicorn server](https://gunicorn.org/).



## Environment variables
Machines running an AIDE service need to have two environment variables set:
* `AIDE_CONFIG_PATH`: location (relative or absolute path) to the [configuration *.ini file](configure_settings.md) on the current machine.
* `AIDE_MODULES`: comma-separated string defining the type(s) of module(s) to be launched on the current machine. The following keywords (case-insensitive) are supported:
    - `LabelUI`: launches all the front-end, resp. user interface functionality
    - `AIController`: makes this machine the central coordinator of AI model training and inference tasks
    - `AIWorker`: makes this machine a model trainer / predictor that receives tasks from the `AIController`
    - `FileServer`: launches the image file server for all projects on this machine
    
    Notes:
    * In standard setups, only the `AIWorker` can be launched on multiple machines natively. However, AIDE should support third-party solutions, such as load balancers, that provide a single access point / URL for multiple machines, which is crucial for all the other services. This has not (yet) been tested, though.
    * The `LabelUI`, `AIController`, and `FileServer` instances' URLs should correspond to the settings provided in the configuration .ini file.
    * Multiple modules can be run on one machine. To do so, just concatenate the module names in a comma-separated list (without white spaces) as an argument for the environment variable.
    * The database is launched separately as a PostGres service.


Setting these environment variables can be done temporarily (example):
```bash
    export AIDE_CONFIG_PATH=config/settings.ini
    export AIDE_MODULES=LabelUI,AIController,FileServer,AIWorker
```

Or permanently (requires re-login):
```bash
    echo "export AIDE_CONFIG_PATH=config/settings.ini" | tee ~/.profile
    echo "export AIDE_MODULES=LabelUI,AIController,FileServer,AIWorker" | tee ~/.profile
```



## Launching AIDE
To launch AIDE (or parts of it, depending on the environment variables set) on the current machine:
```bash
    cd /path/to/your/AIDE/installation
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini
    export AIDE_MODULES=LabelUI
    export PYTHONPATH=.     # might be required depending on your Python setup

    ./AIDE.sh start
```

This launches the Gunicorn HTTP web server, and/or a Celery message broker consumer, depending on the `AIDE_MODULES` environment variable set:

| Module | HTTP web server | Celery |
|--------------|-----------------|--------|
| LabelUI | ✓ | ✓ |
| AIController | ✓ | ✓ |
| AIWorker |  | ✓ |
| FileServer | ✓ | ✓ |


To stop AIDE, simply press Ctrl+C in the running shell. From another shell, you may instead also execute the following command from the root of AIDE, with the correct environment variables set (see above):
```
    ./AIDE.sh stop
```

Note that this stops any Gunicorn process, even if not related to AIDE.
If, for some reason, this fails, the processes can be forcefully stopped manually:
```
    pkill -f celery;
    pkill -f gunicorn;
```