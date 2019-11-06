export AIDE_CONFIG_PATH=serengeti/serengeti-traps.ini



dbName=$(python util/configDef.py --section=Database --parameter=name)
dbUser=$(python util/configDef.py --section=Database --parameter=user)
dbPassword=$(python util/configDef.py --section=Database --parameter=password)
dbPort=$(python util/configDef.py --section=Database --parameter=port)

sudo -u postgres dropdb $dbName

sudo -u postgres psql -c "CREATE DATABASE $dbName WITH OWNER $dbUser CONNECTION LIMIT -1;"
sudo -u postgres psql -c "GRANT CONNECT ON DATABASE $dbName TO $dbUser;"
sudo -u postgres psql -d $dbName -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

# NOTE: needs to be run after init
sudo -u postgres psql -d $dbName -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $dbUser;"

python projectCreation/setupDB.py
python projectCreation/createUsers.py
python projectCreation/import_serengeti_dataset.py --label_folder=serengeti/
