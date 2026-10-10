---
applyTo: "v3python/tune/**/*.py",".tune/**/*.py"
---

These are internal-only scripts for the AOTriton tuning infrastructure.
They are not a public API, are not shipped to end users, and are not run
with `python -O` (we never strip asserts).

Reviewers (including Copilot) MUST NOT:

- Flag `assert` used to validate arguments of internal helpers as a bug.
  `assert` is the preferred form here precisely because it is loud,
  uncaught, and stops execution immediately — unlike `raise ValueError`
  which is routinely swallowed by broad `except Exception:` handlers in
  task runners, Flask error pages, and Celery workers in this codebase.
- Recommend hardening these scripts as if they were a public library
  (e.g. "validate user input", "raise typed exceptions", "add retry/
  timeout for robustness", "handle the `-O` case"). The scope is
  internal tooling; do not extrapolate.
