#!/bin/bash

FILE=/home/aide/files/setup_complete.txt

if [ -f "$FILE" ]; then
    echo "SETUP ALREADY COMPLETED"
else
# =================================
# = FIRST TIME SETUP POSTGRESS DB =
# =================================
    dbName=$(python util/configDef.py --section=Database --parameter=name) \
    && dbUser=$(python util/configDef.py --section=Database --parameter=user) \
    && dbPassword=$(python util/configDef.py --section=Database --parameter=password) \
    && sudo service postgresql restart \
    && sudo -u postgres psql -c "CREATE USER $dbUser WITH PASSWORD '$dbPassword';" \
    && sudo -u postgres psql -c "CREATE DATABASE $dbName WITH OWNER $dbUser CONNECTION LIMIT -1;" \
    && sudo -u postgres psql -c "GRANT CONNECT ON DATABASE $dbName TO $dbUser;" \
    && sudo -u postgres psql -d $dbName -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" \
    && sudo -u postgres psql -d $dbName -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $dbUser;"

# Create DB schema
    python setup/setupDB.py

# =================================
# =   FIRST TIME SETUP RABBITMQ   =
# =================================
    # I need to set rabitmq user and permissions here, as it takes hostname (dynamic) during build of previous phases as part of config folder :-()
    RMQ_username=aide
    RMQ_password=password # This should never be left here for any serious use of course
    sudo service rabbitmq-server start
    # add the user we defined above
    sudo rabbitmqctl add_user $RMQ_username $RMQ_password
    # add new virtual host
    sudo rabbitmqctl add_vhost aide_vhost
    # set permissions
    sudo rabbitmqctl set_permissions -p aide_vhost $RMQ_username ".*" ".*" ".*"

    # Create file to avoid running this script again
    touch $FILE
    echo "FIRST TIME SETUP COMPLETED"
fi
