# Launching AIde

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
The front-end modules of AIde (_LabelUI_, _AIController_, _FileServer_) can be run by invoking the `runserver.py` file with the correct arguments.

To launch just the frontend:
```bash
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini
    python runserver.py --instance=LabelUI
```

Instance arguments can be chained with commas. For example, to launch the _LabelUI_, _AIController_ and _FileServer_ on the same machine, replace the last line with:
```bash
    python runserver.py --instance=LabelUI,AIController,FileServer
```


## AIWorker
_AIWorker_ modules need to be launched using Celery:
```bash
    conda activate aide
    export AIDE_CONFIG_PATH=config/settings.ini
    celery -A modules.AIController.backend.celery_interface worker
```



Note that these instructions launch AIde using Python's built-in WSGI server, which might be detrimental and is not designed for deployment.
To deploy AIde properly, see [here](deployment.md).