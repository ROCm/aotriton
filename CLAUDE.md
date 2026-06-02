# Project Instructions for Claude Code

## Python Version

This project targets **Python 3.10 or newer** (dev environment is Python 3.12). Do not use syntax or features deprecated as of Python 3.10:
- Use `X | Y` union syntax instead of `typing.Union[X, Y]`
- Use `X | None` instead of `typing.Optional[X]`
- Use `list`, `dict`, `tuple` etc. directly as generic types instead of `typing.List`, `typing.Dict`, `typing.Tuple`
- Do not import `Union`, `Optional`, `List`, `Dict`, `Tuple` from `typing` for type annotations

## Python argparse naming

Use **underscores** as word separators for all `argparse` argument names, not
dashes. For example:

```python
# GOOD
parser.add_argument('--sanity_check_only')
parser.add_argument('--database_file')

# BAD
parser.add_argument('--sanity-check-only')
parser.add_argument('--database-file')

Dashes in argument names create attributes with hyphens that cannot be
accessed as args.sanity-check-only; Python silently converts them to
underscores anyway, so be explicit and consistent from the start.
```

## PostgreSQL Connection Configuration

**CRITICAL: NEVER ADD ANY DATABASE NAME TO THE CONFIGURATION OF CONNECTION TO PGSQL. WE NEVER USE IT AND YOU MUST USE THE DEFAULT.**

When creating or modifying PostgreSQL connection parameters:
- DO NOT include `dbname` field in connection parameter dictionaries
- DO NOT add database name to psycopg.connect() calls
- Let psycopg use the default database (same as username)
- Connection params should only contain: `host`, `port`, `user`, `password`

## Database Access Pattern (pq Package)

The `v3python/tune/pq/` package is the **centralized database access library** for all database operations. This ensures CRUD operations are consistent with the schema.

### Key Principles:

1. **pq handles database reads/writes, NOT connections**
   - All functions in `pq/` accept a database connection object (`conn`), not connection parameters
   - Connection management is the caller's responsibility
   - Connections are created once per worker/process and reused

2. **Use pq functions for all database operations**
   - `pq/queue.py` - TaskQueue class for task_queue table operations (fetch, mark_completed, mark_failed, etc.)
   - `pq/results.py` - Functions for tuning_results table (save_tuning_result, get_task_results)
   - DO NOT write raw SQL in other packages (e.g., localq) - use pq functions instead

3. **Pattern Example:**

```python
# GOOD - Persistent connection per worker, use pq functions
import psycopg
from v3python.tune.pq.queue import TaskQueue
from v3python.tune.pq.results import save_tuning_result

class MyWorker:
    def __init__(self, conn_params):
        # Create connection once per worker process
        self.db_conn = psycopg.connect(**conn_params, autocommit=True)
    
    def run(self):
        # Reuse connection for all operations
        task_queue = TaskQueue(self.db_conn)
        
        while self.running:
            tasks = task_queue.fetch_tasks('gfx942', batch_size=10)
            for task in tasks:
                # ... process task ...
                save_tuning_result(task.id, report, self.db_conn)
                task_queue.mark_completed(task.id, arch='gfx942')
    
    def shutdown(self):
        # Close connection on worker shutdown
        if self.db_conn:
            self.db_conn.close()
```

```python
# BAD - Creates new connection per operation
from v3python.tune.pq.queue import TaskQueue

# Don't do this - creates connection internally
task_queue = TaskQueue(conn_params)  # WRONG
```

```python
# BAD - Raw SQL in application code
with db_conn.cursor() as cur:
    cur.execute("""
        UPDATE task_queue
        SET status = 'completed'
        WHERE id = %s
    """, (task_id,))
# Should use: task_queue.mark_completed(task_id, arch)
```

### When to Add Functions to pq:

If you need to perform a database operation:
1. Check if a function exists in `pq/queue.py` or `pq/results.py`
2. If not, add a new function to the appropriate pq module
3. DO NOT write raw SQL in other packages

This keeps all database operations centralized and schema-consistent.

## v3python Package Architecture

The `v3python` package has a strict layering between **description** and **generation**:

- `v3python/base/`, `v3python/kernel/`, `v3python/op/`, `v3python/affine/` — description layer. These serve as structured knowledge for describing kernels and operators, analogous to an AST in a compiler. They should have **minimal knowledge about the code generator**.
- `v3python/codegen/` — generation layer. This drives the code generation process and is the only layer that should contain codegen logic.

Knowledge must not flow upward: description-layer classes must not import from `codegen/` or embed generation logic. When adding a property or method to `Interface`, `KernelDescription`, `Operator`, or the affine classes, ask whether the concept belongs to the *description* of the kernel/operator (fine) or to the *generation process* (put it in `codegen/` instead).

## config.rc Usage

**IMPORTANT: `<workdir>/config.rc` is for AOTriton tuning project configuration ONLY.**

The `config.rc` file is:
- Written in bash syntax for easy consumption by bash scripts
- Used to configure tuning infrastructure (database, workers, containers)
- NOT a general bash/environment configuration file

**DO add to config.rc** (see `config-example.rc`):
- PostgreSQL configuration: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`, `POSTGRES_DOCKER_IMAGE`, `POSTGRES_DOCKER_VOLUME`
- Worker/container configuration: `CONTAINER_SUFFIX`, `CELERY_WORKER_IMAGE`, `CELERY_WORKER_PYTHON`
- Service hosts: `CELERY_SERVICE_HOST`
- SLURM configuration: `SLURM_LOGIN_NODE`, `SLURM_WORKER_DIR`, `SLURM_MODULES`

**DO NOT add to config.rc:**
- General environment variables: `PYTHONDONTWRITEBYTECODE`, `PYCACHEPREFIX`, `PATH`, `LD_LIBRARY_PATH`
- System-wide shell configuration
- Python runtime settings
- Non-tuning infrastructure settings

**For general environment configuration**, use:
- Docker environment variables: `docker run -e PYCACHEPREFIX=/tmp/pycache`
- Worker startup scripts: `.tune/remote/worker_service.sh`
- Container Dockerfiles or image configuration
