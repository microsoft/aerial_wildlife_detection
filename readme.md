# AIde - Assisted Interface that does everything

AIde is a modular, multi-tier web framework for labeling image datasets with AI assistance and Active Learning support.


## Overview

TODO


## Installation

TODO: steps:
1. Install required software (TODO: write setup.py file, eventually provide docker script?)
2. Provide parameters in `/config` directory with project-specific settings (database, interface parameters, etc.)
3. Launch script that sets up project (TODO: write bash script in projectSetup folder)
4. Populate database with data
5. Launch the desired instance (see below)


### Set up the database instance

AIde uses [PostgreSQL](https://www.postgresql.org/) to store labels, predictions, file paths and metadata. The following instructions apply for recent versions of Debian-based Linux distributions, such as Ubuntu.
Note that AIde requires PostgreSQL >= 9.5.

*Installing and configuring PostgreSQL*
1. Install PostgreSQL server
```
    sudo apt-get update && sudo apt-get install -y wget
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
    sudo apt-get update && sudo apt-get install -y postgresql-10

    //TODO: change peer to md5 in /etc/postgresql/10/main/pg_hba.conf

    //TODO
    sudo sed -e "s/#[ ]*listen_addresses = 'localhost'/listen_addresses = '\*'/g" /etc/postgresql/10/main/postgresql.conf

    //TODO: also need to replace the local IP address with "all" in pg_hba.conf (EXTREMELY UNSAFE)
    sudo echo "host    all             all             0.0.0.0/0               md5" >> /etc/postgresql/10/main/pg_hba.conf

    sudo service postgresql restart
    sudo systemctl enable postgresql
```

2. Create a new database and the main user account. This needs to be done from the installation root of the AIlabelTool,
   with the correct environment activated.
```
    sudo -u postgres psql -c "CREATE USER $(python util/configDef.py --section=Database --parameter=user) WITH PASSWORD '$(python util/configDef.py --section=Database --parameter=user)';"
    sudo -u postgres psql -c "CREATE DATABASE $(python util/configDef.py --section=Database --parameter=name) WITH OWNER $(python util/configDef.py --section=Database --parameter=user) CONNECTION LIMIT -1;"
    sudo -u postgres psql -c "GRANT CONNECT ON DATABASE $(python util/configDef.py --section=Database --parameter=name) TO $(python util/configDef.py --section=Database --parameter=user);"
    sudo -u postgres psql -d $(python util/configDef.py --section=Database --parameter=name) -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

    //TODO: needs to run after init
    sudo -u postgres psql -d (python util/configDef.py --section=Database --parameter=name) -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO (python util/configDef.py --section=Database --parameter=user);"
```

3. Setup the database schema. We do that using the newly created user account instead of the postgres user:
```
    python projectCreation/setupDB.py
```



## Launching an instance

To start an instance, run the "runserver.py" script with the appropriate argument for the desired type of module (i.e., one of: "LabelUI", "AIController", "AIWorker", "FileServer").
Example: the following command would start a labeling UI frontend on the current machine, using the parameters specified in the `config` directory:

`python runserver.py --instance=LabelUI`


It is also possible to run multiple modules on the same instance by providing comma-separated module names as an argument:

`python runserver.py --instance=LabelUI,AIController`