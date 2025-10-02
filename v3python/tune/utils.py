import sys
from dataclasses import fields
import json

def safeload(s):
    return json.loads(s) if s else None

def parse_python(line: str) -> dict:
    args = line.split(';')
    d = {}
    for assignment in args:
        # print(f'{assignment=}', file=sys.stderr)
        k, v = assignment.split('=', maxsplit=1)
        d[k] = eval(v)
    return d

def asdict_shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}
