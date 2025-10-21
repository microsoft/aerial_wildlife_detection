# Manual installation for Debian/Ubuntu

Follow these instructions if you are encountering issues with the [installer](install_overview.md#Debian-) or would like to take full control over which components you want to install.

## Requirements

The AIDE label interface requires the following libraries:

* bottle>=0.12
* psycopg2>=2.8.2
* tqdm>=4.32.1
* bcrypt>=3.1.6
* netifaces>=0.10.9
* gunicorn>=19.9.0
* Pillow>=2.2.1
* numpy>=1.16.4
* requests>=2.22.0
* celery[librabbitmq,redis,auth,msgpack]>=4.3.0


Finally, the [built-in models](builtin_models.md) require:

* pytorch>=1.5.0
* torchvision>=0.6.0
* detectron2>=0.3.0
* opencv-python

If you have a CUDA-capable GPU it is highly recommended to install PyTorch with GPU support (see the [official website](https://pytorch.org/get-started/locally/)).


## Step-by-step installation

The following installation routine had been tested on Ubuntu >= 16.04. AIDE will likely run on different OS as well, with instructions requiring corresponding adaptations.



### Prepare environment

Run the following code snippets on all machines that run one of the services for AIDE (_LabelUI_, _AIController_, _AIWorker_, etc.).
It is strongly recommended to run AIDE in a self-contained Python environment, such as [Conda](https://conda.io/) (recommended and used below) or [Virtualenv](https://virtualenv.pypa.io).

```bash
    # specify the root folder where you wish to install AIDE
    targetDir=/path/to/desired/source/folder

    # create environment (requires conda or miniconda)
    conda create -y -n aide python=3.9
    conda activate aide

    # download AIDE source code
    sudo apt-get update && sudo apt-get install -y git
    cd $targetDir
    git clone https://github.com/microsoft/aerial_wildlife_detection.git

    # install required libraries
    sudo apt-get install -y build-essential libpq-dev python-dev ffmpeg libsm6 libxext6 python3-opencv
    pip install -U -r requirements.txt
```


### Create the settings.ini file

Every instance running one of the services for AIDE gets its general required properties from a *.ini file.
It is highly recommended to prepare a .ini file at the start of the installation of AIDE and to have a copy of the same file on all machines.
Note that in the latest version of AIDE, the .ini file does not contain any project-specific parameters anymore.
**Important: NEVER, EVER make the configuration file accessible to anyone, let alone the network or Web.**

1. Create a *.ini file for your general AIDE setup. See the provided file under `config/settings.ini` for an example. To view all possible parameters, see [here](configure_settings.md).
2. Copy the *.ini file to each server instance.
3. On each instance, set the `AIDE_CONFIG_PATH` environment variable to point to your *.ini file:
```bash
    # temporarily:
    export AIDE_CONFIG_PATH=/path/to/settings.ini

    # permanently (requires re-login):
    echo "export AIDE_CONFIG_PATH=path/to/settings.ini" | tee ~/.profile
```


### Set up the database instance

See [here](setup_db.md)



### Set up the message broker

The message broker is required for all services of AIDE, except for the Database.
To set up the message broker correctly, see [here](installation_aiTrainer.md).





### Import existing data

In the latest version, AIDE offers a GUI solution to configure projects and import and export images.
At the moment, previous data management scripts listed [here](import_data.md) only work if the configuration .ini
file contains all the legacy, project-specific parameters required for the previous version of AIDE.
New API scripts are under development.



### Launch the modules

See [here](launch_aide.md)