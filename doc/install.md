# Installation

## Requirements

The AIde label interface (without the AI backend) requires the following libraries:

* bottle>=0.12
* psycopg2>=2.8.2
* tqdm>=4.32.1
* bcrypt>=3.1.6
* netifaces>=0.10.9

The AI backend core further relies on:

* celery[librabbitmq,redis,auth,msgpack]>=4.3.0


Finally, the [built-in models](builtin_models.md) require:

* numpy>=1.16.4
* pytorch>=1.1.0
* torchvision>=0.3.0

It is highly recommended to install PyTorch with GPU support (see the [official website](https://pytorch.org/get-started/locally/)).


## Step-by-step installation

The following installation routine had been tested on Ubuntu 16.04. AIde will likely run on different OS as well, with instructions requiring corresponding adaptations.



### Prepare environment

Run the following code snippets on all machines that run one of the services for AIde (_LabelUI_, _AIController_, _AIWorker_, etc.).
It is strongly recommended to run AIde in a self-contained Python environment, such as [Conda](https://conda.io/) (recommended and used below) or [Virtualenv](https://virtualenv.pypa.io).

```bash
    # specify the root folder where you wish to install AIde
    targetDir=/path/to/desired/source/folder

    # install required software
    sudo apt-get update && sudo apt-get install -y git

    # create environment (requires conda or miniconda)
    conda create -y -n aide python=3.7
    conda activate aide

    # install basic requirements
    pip install -U -r requirements.txt

    # at this point you may want to install the additonal packages listed above, if required

    # download AIde source code
    cd $targetDir
    git clone git+https://github.com/microsoft/aerial_wildlife_detection.git
```


### Create the settings.ini file

Every instance running one of the services for AIde gets its required properties from a *.ini file.
It is highly recommended to prepare a .ini file at the start of each project and to have a copy of the same file on all machines.
**Important: NEVER, EVER make the configuration file accessible to the outside web.**

1. Create a *.ini file for your project. See the provided file under `config/settings.ini` for an example. To view all possible parameters, see [here](configure_settings.md).
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

If you only want to use AIde as a labeling platform (i.e, without any AI model in the background), you can skip this step. Otherwise see [here](installation_aiTrainer.md).





### Import existing data

Importing images (and labels) into a running database is explained [here](import_data.md).



### Launch the modules

See [here](launch_aide.md)