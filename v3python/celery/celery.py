from celery import Celery

app = Celery('celery_tuner',
             include=['v3python.celery.tasks'])

app.config_from_object('v3python.celery.celeryconfig')

if __name__ == '__main__':
    app.start()
