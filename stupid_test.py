import celery
from kombu import Queue
from kombu.common import Broadcast


app = celery.Celery('stupid_test',
            broker='amqp://aiLabelUser:aiLabelPassword@10.0.0.4:5672/rabbitmq_vhost',        #TODO
            backend='redis://10.0.0.4:6379/0')   #TODO
app.conf.update(
    result_backend='redis://10.0.0.4:6379/0',
    task_ignore_result=False,
    result_persistent=True,
    accept_content = ['json'],
    task_serializer = 'json',
    result_serializer = 'json',
    task_track_started = True,
    broker_pool_limit=None,                 # required to avoid peer connection resets
    broker_heartbeat = 0,                   # required to avoid peer connection resets
    worker_max_tasks_per_child = 1,         # required to free memory (also CUDA) after each process
    task_default_rate_limit = 3,            #TODO
    worker_prefetch_multiplier = 1,         #TODO
    task_acks_late = True,
    task_create_missing_queues = True,
    # task_routes = {'my.function.add': 'stupid_test'},
    # task_default_queue = 'stupid_test',
    task_queues = (Queue('stupid_test'), Queue('testQueue'),)
    # task_queues = (Broadcast('aide_broadcast'), Queue('FileServer'), Queue('AIController'), Queue('AIWorker')),
    # task_routes = {
    #     'aide_admin': {
    #         'queue': 'aide_broadcast',
    #         'exchange': 'aide_broadcast'
    #     }
    # }
)



@app.task(name='my.function.add')
def add(x, y):
    print("Adding {} + {}".format(x, y))
    return x + y



if __name__ == '__main__':
    job = celery.chain(add.s(2, 2).set(queue='testQueue'), add.s(4).set(queue='testQueue'), add.s(8).set(queue='testQueue'))
    result = job.apply_async(queue='testQueue')
    output = result.get()
    print("Result:")
    print(output)