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



## Step-by-step installation

The following installation routine had been tested on Ubuntu 16.04. AIde will likely run on different OS as well, with instructions requiring corresponding adaptations.
For deployment it is imperative to use a proper web server, such as [NGINX](https://www.nginx.com/) (which is the recommended way of deploying AIde, and shown below). However, for debugging purposes you may use Python's built-in server capabilities. Details are shown in the setup snippets below.



### Prepare environment

Run the following code snippets on all machines that run one of the services for AIde (_LabelUI_, _AIController_, _AIWorker_, etc.).
It is strongly recommended to run AIde in a self-contained Python environment, such as [Conda](https://conda.io/) (recommended and used below) or [Virtualenv](https://virtualenv.pypa.io).

```
    # specify the root folder where you wish to install AIde
    targetDir=/path/to/desired/source/folder

    # install required software
    sudo apt-get update && sudo apt-get install -y git

    # create environment
    conda create -y -n aide python=3.7
    conda activate aide

    # install basic requirements
    pip install requirements.txt

    # download AIde source code
    cd $targetDir
    git clone git+https://github.com/microsoft/aerial_wildlife_detection.git
```


### Create the settings.ini file

Every instance running one of the services for AIde gets its required properties from a *.ini file.
It is highly recommended to prepare a .ini file at the start of each project and to have a copy of the same file on all machines.
**<span style="color:red">Important: NEVER, EVER make the configuration file accessible to the outside web.</span>**

1. Create a *.ini file for your project. See the provided file under `config/settings.ini` for an example. To view all possible parameters, see [here](doc/configure_settings.md).
2. Copy the *.ini file to each server instance.
3. On each instance, set the `AIDE_CONFIG_PATH` environment variable to point to your *.ini file:
```
    # temporarily:
    export AIDE_CONFIG_PATH=/path/to/settings.ini

    # permanently (requires re-login):
    echo "export AIDE_CONFIG_PATH=/path/to/settings.ini" > ~/.profile
```


### Set up the database instance

AIde uses [PostgreSQL](https://www.postgresql.org/) to store labels, predictions, file paths and metadata. The following instructions apply for recent versions of Debian-based Linux distributions, such as Ubuntu.
Note that AIde requires PostgreSQL >= 9.5 (it has been tested with version 10).


1. Install PostgreSQL server

```
    # install packages
    sudo apt-get update && sudo apt-get install -y wget
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
    sudo apt-get update && sudo apt-get install -y postgresql-10


    # modify authentication (NOTE: you might want to manually adapt these commands for increased security)
    sudo sed -e "s/#[ ]*listen_addresses = 'localhost'/listen_addresses = '\*'/g" /etc/postgresql/10/main/postgresql.conf
    sudo echo "host    all             all             0.0.0.0/0               md5" >> /etc/postgresql/10/main/pg_hba.conf

    sudo service postgresql restart
    sudo systemctl enable postgresql


    # If AIde is run on MS Azure: TCP connections are dropped after 4 minutes of inactivity
    # (see https://docs.microsoft.com/en-us/azure/load-balancer/load-balancer-outbound-connections#idletimeout)
    # This is fatal for our database connection system, which keeps connections open.
    # To avoid idling/dead connections, we thus use Ubuntu's keepalive timer:
    echo "net.ipv4.tcp_keepalive_time = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_intvl = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_probes = 20" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    sudo sysctl -p
```


2. Define database details

```
    dbName=$(python util/configDef.py --section=Database --parameter=name)
    dbUser=$(python util/configDef.py --section=Database --parameter=user)
    dbPassword=$(python util/configDef.py --section=Database --parameter=user)
```


3. Create a new database and the main user account. This needs to be done from the installation root of AIde,
   with the correct environment activated.

```
    sudo -u postgres psql -c "CREATE USER $dbUser WITH PASSWORD '$dbPassword';"
    sudo -u postgres psql -c "CREATE DATABASE $dbName WITH OWNER $dbUser CONNECTION LIMIT -1;"
    sudo -u postgres psql -c "GRANT CONNECT ON DATABASE $dbName TO $dbUser;"
    sudo -u postgres psql -d $dbName -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

    # NOTE: needs to be run after init
    sudo -u postgres psql -d $dbName -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $dbUser;"
```


4. Setup the database schema. We do that using the newly created user account instead of the postgres user:

```
    python projectCreation/setupDB.py
```


5. If you wish you can remove the AIde code base (and Python environment) from the database server now, unless the server hosts any other AIde module.




### Set up the _LabelUI_ frontend

