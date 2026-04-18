# Project Instructions for Claude Code

## PostgreSQL Connection Configuration

**CRITICAL: NEVER ADD ANY DATABASE NAME TO THE CONFIGURATION OF CONNECTION TO PGSQL. WE NEVER USE IT AND YOU MUST USE THE DEFAULT.**

When creating or modifying PostgreSQL connection parameters:
- DO NOT include `dbname` field in connection parameter dictionaries
- DO NOT add database name to psycopg.connect() calls
- Let psycopg use the default database (same as username)
- Connection params should only contain: `host`, `port`, `user`, `password`
