export AIDE_CONFIG_PATH=bison/bison.ini



dbName=$(python util/configDef.py --section=Database --parameter=name)
dbUser=$(python util/configDef.py --section=Database --parameter=user)
dbPassword=$(python util/configDef.py --section=Database --parameter=password)
dbPort=$(python util/configDef.py --section=Database --parameter=port)

# specify postgres version you wish to use (must be >= 9.5)
version=10


# install packages
sudo apt-get update && sudo apt-get install -y wget
echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y postgresql-$version


# update the postgres configuration with the correct port
sudo sed -i "s/\s*port\s*=\s[0-9]*/port = $dbPort/g" /etc/postgresql/$version/main/postgresql.conf


# modify authentication
# NOTE: you might want to manually adapt these commands for increased security; the following makes postgres listen to all global connections
sudo sed -i "s/\s*#\s*listen_addresses\s=\s'localhost'/listen_addresses = '\*'/g" /etc/postgresql/$version/main/postgresql.conf
echo "host    all             all             0.0.0.0/0               md5" | sudo tee -a /etc/postgresql/$version/main/pg_hba.conf > /dev/null


# restart postgres and auto-launch it on boot
sudo service postgresql restart
sudo systemctl enable postgresql


# If AIde is run on MS Azure: TCP connections are dropped after 4 minutes of inactivity
# (see https://docs.microsoft.com/en-us/azure/load-balancer/load-balancer-outbound-connections#idletimeout)
# This is fatal for our database connection system, which keeps connections open.
# To avoid idling/dead connections, we thus use Ubuntu's keepalive timer:
if ! sudo grep -q ^net.ipv4.tcp_keepalive_* /etc/sysctl.conf ; then
    echo "net.ipv4.tcp_keepalive_time = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_intvl = 60" | sudo tee -a "/etc/sysctl.conf" > /dev/null
    echo "net.ipv4.tcp_keepalive_probes = 20" | sudo tee -a "/etc/sysctl.conf" > /dev/null
else
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_time.*/net.ipv4.tcp_keepalive_time = 60 /g" /etc/sysctl.conf
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_intvl.*/net.ipv4.tcp_keepalive_intvl = 60 /g" /etc/sysctl.conf
    sudo sed -i "s/^\s*net.ipv4.tcp_keepalive_probes.*/net.ipv4.tcp_keepalive_probes = 20 /g" /etc/sysctl.conf
fi
sudo sysctl -p
