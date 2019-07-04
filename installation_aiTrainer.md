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
    sudo rabbitmqctl add_user <user> <password>

    # add new virtual host
    sudo rabbitmqctl add_vhost <vhost_name>

    # set permissions for user on vhost
    sudo rabbitmqctl set_permissions -p <vhost_name> <user> ".*" ".*" ".*"

    # restart rabbit
    sudo rabbitmqctl restart
```