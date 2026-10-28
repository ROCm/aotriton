# Project Instructions for Claude Code

## Product Naming

Name products by their architecture identifier only (e.g. gfx1250, gfx942).
Do not add commercial names, code names, or any other aliases. Only manual
editing by the user may introduce alternative names.

## Do Not Name Things "kind"

Do not use `kind` in any identifier or term: variables, fields, columns,
arguments, CLI flags, classes, or concepts in design documents. That includes
compounds such as `subkind`, `runner_kind` and `task_kind`.

The one exception is code that is closely tied to a compiler concept, or that is
itself a component of a compiler. There, `kind` is established vocabulary (AST
node kinds, token kinds). Outside compilers the word is rare and says nothing
about what it names. Pick a word that describes the actual distinction, such
as `type`, `role`, `level` or `mode`, or a domain-specific term.

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

## Database Access Pattern (pq Packages)

**The main goal: raw PostgreSQL statements must not exist outside a `pq`
package.** Everything below serves that goal.

Each application or library has exactly one (sub-)package named `pq` that
handles all of its PostgreSQL access. It holds that layer's schema and every
query.

For example, the distributed execution engine has `pgdef/pq` for its own tables,
and each application (tune, and later perfmon) has its own `pq` for theirs. A
layer's `pq` may *read* another layer's tables directly (for example, to join
against them), but it *writes* them only by calling that layer's `pq`
functions. Locations change over the project's history (for example
`v3python/tune/pq/`, later `python/tune/pq/`), so locate the current `pq`
packages instead of assuming paths. The rule does not change.

### Key Principles:

1. **No SQL outside pq.** Application code, workers, scripts and tools call pq
   functions. If the operation you need does not exist, add a function to the
   `pq` of the layer that owns the table. Do not write SQL at the call site.

2. **pq handles reads and writes, NOT connections.**
   - pq functions take a connection object (`conn`), never connection
     parameters.
   - The caller creates one connection per worker or process, and reuses it.
   - Connection parameters follow the PostgreSQL section above (no `dbname`).

3. **Pattern Example** (function names are illustrative):

```python
# GOOD - one connection per process, all SQL behind pq functions
import psycopg
from <pq package> import tasks

class MyWorker:
    def __init__(self, conn_params):
        self.db_conn = psycopg.connect(**conn_params, autocommit=True)

    def run(self):
        while self.running:
            for task in tasks.claim(self.db_conn, ...):
                # ... process task ...
                tasks.complete(self.db_conn, task.id, ...)

    def shutdown(self):
        if self.db_conn:
            self.db_conn.close()
```

```python
# BAD - pq object that opens its own connection
queue = TaskQueue(conn_params)  # WRONG: pass a conn

# BAD - raw SQL outside pq
with db_conn.cursor() as cur:
    cur.execute("UPDATE ... SET status = 'completed' WHERE id = %s", (task_id,))
```

This keeps each layer's database operations in one place, consistent with its
schema, and relocatable as a unit.

## v3python Package Architecture

The `v3python` package has a strict layering between **description** and **generation**:

- `v3python/base/`, `v3python/kernel/`, `v3python/op/`, `v3python/affine/` — description layer. These serve as structured knowledge for describing kernels and operators, analogous to an AST in a compiler. They should have **minimal knowledge about the code generator**.
- `v3python/codegen/` — generation layer. This drives the code generation process and is the only layer that should contain codegen logic.

Knowledge must not flow upward: description-layer classes must not import from `codegen/` or embed generation logic. When adding a property or method to `Interface`, `KernelDescription`, `Operator`, or the affine classes, ask whether the concept belongs to the *description* of the kernel/operator (fine) or to the *generation process* (put it in `codegen/` instead).

## Keep the External Tuning Database Option

A tuning database is part of its module, at `<module>/database` (e.g.
`modules/flash/database`). The **external tuning database** option lets a build
use a database from outside the modules instead:

- `.ci/build-*.sh --external_tuning_database <dir>`
- CMake: `-DAOTRITON_EXTERNAL_TUNING_DATABASE=<dir>`
- `aotriton.generate` / `aotriton.write_image_signature`:
  `--external_tuning_database <dir>`

`<dir>/<family>/database` replaces `<module>/database` for each family present
in `<dir>`; other modules keep their own database. It originated in Tuner v3.5
(build with a database freshly exported from tuning to evaluate it before
committing it), but it is not limited to the tuning framework: anyone who wants
to build with a different tuning database uses it.

**Do not remove this option, and do not fold it into the per-module
database lookup**, even though nothing in the default build uses it. It has
been removed by mistake before (it was then called `--database_root`).

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

## .gitignore Changes Require Human Review

**NEVER add entries to `.gitignore` on your own initiative.** Propose the change and
let the user decide.

An ignore entry does not just keep a file out of a commit — it removes the file from
`git status` entirely. Files sitting untracked in the working tree are often there on
purpose and still need attention: drafted PR descriptions awaiting backup out of the
repo, downloaded reference material, scratch output not yet filed. Silencing them
means they are forgotten rather than handled. That trade-off is the user's to make,
never yours.

This applies even when an ignore rule looks like the obvious fix for a problem you
just caused. If a file was committed by accident, the fix is to stage deliberately,
not to make the file invisible:

```bash
# BAD - sweeps up every untracked file in the tree, then hides the evidence
git add -A
echo '/*.md' >> .gitignore

# GOOD - name what you are committing
git add path/to/file.py path/to/other.py
```

**Do not use `git add -A` or `git add .`.** Stage explicit paths. Blanket staging is
what pulls stray downloads and scratch files into commits in the first place.

Removing or narrowing an existing `.gitignore` entry is subject to the same rule:
suggest it, explain what would become visible or tracked, and wait.
