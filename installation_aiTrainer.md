#TODO

To prepare a machine to run the AIController instance:

```
    # install RabbitMQ server
    sudo apt-get install -y rabbitmq-server

    # install Celery
    pip install celery
```


To set up a machine as an AIWorker:
(TODO: source: https://avilpage.com/2014/11/scaling-celery-sending-tasks-to-remote.html)
```
    # add new user
    sudo rabbitmqctl add_user aiLabelUser aiLabelPassword

    # add new virtual host
    sudo rabbitmqctl add_vhost rabbitmq_vhost

    # set permissions for user on vhost
    sudo rabbitmqctl set_permissions -p rabbitmq_vhost aiLabelUser ".*" ".*" ".*"

    # restart rabbit
    sudo service rabbitmq-server stop       # may take a minute; if the command hangs: pkill -KILL -u rabbitmq
    sudo service rabbitmq-server start
```