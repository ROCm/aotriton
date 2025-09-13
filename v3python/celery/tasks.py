from .celery import app
from celery.signals import celeryd_after_setup, celeryd_init
from celery import current_task
import os
import logging
import sys

@app.task
def add(x, y):
    return x + y

@app.task
def mul(x, y):
    return x * y

@app.task
def xsum(numbers):
    return sum(numbers)

@app.task
def get_worker_name_task():
    worker_hostname = current_task.request.hostname
    return worker_hostname

def route_task(name, args, kwargs, options, task=None, **kw):
    arch = kwargs['arch']
    d = {
        'queue': arch,
        'routing_key': f'{arch}.{name}',
    }
    return d
