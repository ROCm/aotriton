# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The task category: a short path saying which application runs a task and,
optionally, which type of worker within it.

    PostgreSQL  one column per level: cat_level0 TEXT NOT NULL, cat_level1 TEXT
    Python      tuple[str, str | None]   e.g. ('tune', 'kernel'), ('tune', None)
    CLI / shell --category tune,kernel   or   --category tune

Conversion between the three happens only here and in the other pq modules.

The pair rule, shared with (pool, subpool): a NULL level on a TASK means
"generic"; a reader configured with a level claims tasks whose level is NULL or
equal to its own. Only trailing levels may be NULL.

To go N-dimensional, add a `cat_levelN` column, raise LEVELS, and extend the
CHECK in schema.sql: claim_terms() generates one term per level.
"""

import re

Category = tuple[str, str | None]

LEVELS = 2

_LEVEL_RE = re.compile(r'^[A-Za-z0-9_]+$')


def _check_level(text: str, what: str) -> str:
    if not isinstance(text, str) or not _LEVEL_RE.match(text):
        raise ValueError(f"invalid category {what} {text!r}: expected [A-Za-z0-9_]+")
    return text


def make_category(level0: str, level1: str | None = None) -> Category:
    return (_check_level(level0, 'level 0'),
            None if level1 is None else _check_level(level1, 'level 1'))


def parse_category(text: str) -> Category:
    """'tune,kernel' -> ('tune', 'kernel'); 'tune' -> ('tune', None)."""
    parts = text.split(',')
    if len(parts) > LEVELS:
        raise ValueError(f"category {text!r} has {len(parts)} levels; at most {LEVELS}")
    parts = [p.strip() for p in parts]
    return make_category(*parts)


def format_category(category: Category) -> str:
    """The inverse of parse_category()."""
    return ','.join(level for level in category if level is not None)


def to_columns(category: Category) -> dict[str, str | None]:
    """{'cat_level0': ..., 'cat_level1': ...}, validated; for SQL parameters."""
    if len(category) > LEVELS:
        raise ValueError(f"category {category!r} has {len(category)} levels; at most {LEVELS}")
    level0, level1 = make_category(*category)
    return {'cat_level0': level0, 'cat_level1': level1}


def from_columns(row) -> Category:
    """Build a Category from a mapping holding cat_level0 and cat_level1."""
    return (row['cat_level0'], row['cat_level1'])


def claim_terms(levels: int = LEVELS, *, alias: str = '') -> str:
    """The category part of a claim's WHERE clause, as SQL text with named
    parameters %(cat_level0)s, %(cat_level1)s, ...

    Level 0 is an equality (a reader always serves exactly one application).
    Every deeper level is the pair rule.
    """
    pfx = f'{alias}.' if alias else ''
    terms = [f'{pfx}cat_level0 = %(cat_level0)s']
    for n in range(1, levels):
        terms.append(f'({pfx}cat_level{n} IS NULL OR {pfx}cat_level{n} = %(cat_level{n})s)')
    return ' AND '.join(terms)


def argparse_type(text: str) -> Category:
    """`type=` for argparse: turns a bad value into a usage error."""
    import argparse
    try:
        return parse_category(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
