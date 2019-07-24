# Deploying AIde

The following instructions deploy AIde using the dedicated web server [NGINX](https://www.nginx.com/).
This affects the following instances:
* The _LabelUI_ module;
* The _AIController_ module;
* The file server.

The commands below retrieve parameters from the *.ini configuration file on the respective instance and thus require to be run in the AIde root and with the `AIDE_CONFIG_PATH` environment variable set properly.



## Install NGINX

Carry out these instructions for all machines running one of the modules listed above.
There is no need to use a web server for the _AIWorker_ instances, as these communicate using a message broker directly.

```
    # install nginx
    release=$(lsb_release -sc)
    echo "deb http://nginx.org/packages/ubuntu/ $release nginx" | sudo tee -a "/etc/apt/sources.list.d/nginx.list" >> /dev/null
    echo "deb-src http://nginx.org/packages/ubuntu/ $release nginx" | sudo tee -a "/etc/apt/sources.list.d/nginx.list" >> /dev/null
    sudo apt-get update && sudo apt-get install nginx
```


## Configure the _LabelUI_ and/or _AIController_ modules

The following steps are required for every instance running one of the primary AIde modules.
If an instance is supposed to run multiple modules at once (e.g. both the _LabelUI_ and _AIController_), provide both names comma-separated below.


```
    envDir=/path/to/python/environment/bin/python                           # absolute path of the Python executable. Example: "/data/anaconda/envs/py36/bin/python". See command: "conda info --envs".
    aideDir=/path/to/aide/root                                              # absolute directory where the AIde code base is installed in
    configFilePath=/path/to/settings.ini                                    # absolute directory where the project's *.ini configuration file lies
    modules='LabelUI,AIController'                                          # modules to be run on this very server. You may add multiple as a comma-separated string (the order does not matter)
    port=$(python util/configDef.py --section=Server --parameter=port);     # port under which you wish to run the _LabelUI_ frontend
    user=$whoami;                                                           # user name under which to run the AIde instance
    numWorkers=5;                                                           # number of threads to run the Gunicorn server in


    # set up gunicorn service
    echo "
    [Unit]
    Description=Gunicorn instance for AIde
    After=network.target

    [Service]
    User=$user
    Group=#TODO
    WorkingDirectory=$aideDIR
    Environment=AIDE_CONFIG_PATH=$configFilePath
    Environment=PATH=
    ExecStart=gunicorn --workers $numWorkers --bind unix:aide.sock -m 007 application:app


    [Install]
    WantedBy=multi-user.target
    " | sudo tee -a "/etc/systemd/system/aide.service" 


    # configure nginx proxy
    echo "
        server {
            listen $port;
        }
        #TODO: proxy_pass
    " | sudo tee -a "/etc/nginx/sites-available/aide" >> /dev/null

    sudo ln -s /etc/nginx/sites-available/aide /etc/nginx/sites-enabled

    # start service
    sudo systemctl restart nginx
```


## Configure the file server

```
    port=$(python util/configDef.py --section=Server --parameter=port);              # port under which you wish to run the file server
    fileDir=$(python util/configDef.py --section=FileServer --parameter=staticfiles_dir);      # absolute path under which the images are stored
    webDir=$(python util/configDef.py --section=FileServer --parameter=staticfiles_uri);          # web path under which the images can be retrieved
    echo "server {
        listen     $port default_server;

        location $webDir {
            alias $fileDir;
            autoindex off;
            sendfile on;
            sendfile_max_chunk 1m;
        }
    }" | sudo tee -a "/etc/nginx/conf.d/aide.conf" >> /dev/null


    # start service
    sudo systemctl restart nginx
```