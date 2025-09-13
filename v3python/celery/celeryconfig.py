class ConfigRC(object):
    def __init__(self):
        self.broker_url = 'amqp://guest:guest@localhost:5672'
        self.result_backend = 'rpc://'
        import os
        workdir = os.getenv('AOTRITON_CELERY_WORKDIR', None)
        if workdir is None:
            return
        from argparse import Namespace
        args = Namespace()
        from pathlib import Path
        with open(Path(workdir) / 'config.rc') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                k, v = line.split('#')[:2]
                setattr(args, k, v)
        self.broker_url = 'amqp://{}:{}@{}:{}//'.format(args.RABBITMQ_DEFAULT_USER,
                                                         args.RABBITMQ_DEFAULT_PASS,
                                                         args.CELERY_SERVICE_HOST,
                                                         args.RABBITMQ_NODE_PORT)
        self.result_backend = 'db+postgresql://{}:{}@{}:{}'.format(args.POSTGRES_USER,
                                                                   args.POSTGRES_PASSWORD,
                                                                   args.CELERY_SERVICE_HOST,
                                                                   args.POSTGRES_PORT)


rc = ConfigRC()
broker_url = rc.broker_url
result_backend = rc.result_backend
worker_concurrency = 16
# task_routes = ('v3python.celery.tasks.route_task', )
