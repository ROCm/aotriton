#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: psql.sh <dir>' >&2
  exit 1
fi

dir="$1"
rcfile="$dir/config.rc"
. "$rcfile"

PSQL_CONNECTION_STRING="postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$CELERY_SERVICE_HOST:$POSTGRES_PORT"

docker run -ti --rm alpine/psql:17.6 $PSQL_CONNECTION_STRING

