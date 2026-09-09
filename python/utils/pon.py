# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PON (Plain / Python Object Notation): a safe, separator-configurable
`k=v;k=v` wire format, and the one home for both reading and writing it.

`aotriton.tune.utils.parse_python` -- the format's original reader -- splits a
line on `;`, then on `=` with `maxsplit=1`, then calls `eval(v)` on the value.
Two problems: the `eval` can execute arbitrary code, and the separator is
hard-coded to `;`, so a caller whose wire separator is something else (a
space-separated `--signature`/`--hints` command line, say) cannot reuse it and
writes its own splitter instead.

`parse_pon` keeps the shape and fixes both: `ast.literal_eval` accepts exactly
the forms build inputs use -- ints, floats, quoted strings, tuples, lists,
`True`/`False`/`None` -- and raises on anything else, including bare
identifiers. `sep` defaults to `';'` so it is a drop-in for every
`parse_python` caller (`FlashEntry.parse_text`,
`FlashInputMetadata.parse_text`); a caller with a different wire separator
passes `sep=' '` and gets the same parser rather than a second spelling of it.

`render_pon` is the matching writer: `repr()`-compatible per value, not
`str()`, so every value it emits is exactly what `ast.literal_eval` (hence
`parse_pon`) would read back -- the format is round-trippable in both
directions, and there is only one dialect. For `str` values specifically,
`render_pon` requires `repr(v) == "'" + v + "'"`: single-quoted, nothing
escaped. If a value ever stops satisfying that (a quote, backslash, or
non-printable character in it), it raises loudly, naming the key, at the point
the wire text is built -- rather than silently emitting text that only a full
Python-repr unescaper could read back. Holding the grammar down to that is what
lets a reader outside Python strip exactly one pair of surrounding single
quotes and be done.

**A PON string contains NO SPACES.** This is a hard property of the format, not
a cosmetic one, and it is why `render_pon` cannot simply call `repr()` on a
container: `repr((1, 2))` is `'(1, 2)'`, with a space after the comma.

The reason is the exaid/testrun wire protocol. `ExaidProxy.write`
(`python/tune/exaid.py`) joins its arguments with `' '`, and the worker's
`first()` (`python/tune/testrun.py`) splits the line back apart on `' '`. A
single embedded space therefore does not corrupt the value -- it silently
re-tokenizes the whole command, so a later positional argument is read from the
middle of an earlier one. `render_pon` renders containers with a bare `,`
separator and rejects any value whose rendering would contain a space,
including a `str` with a space inside it.

**What `render_pon` rejects, and why the list is what it is.** The round-trip
promise above -- `parse_pon(render_pon(d)) == d` -- is not a property of the
grammar, it is a property of the writer refusing everything the grammar cannot
carry. Three of those refusals are about text that would parse *without error*
into a different dict, which is the failure mode worth being loud about:

* a `sep` that is one of `,[]()='"\\` -- the characters this writer emits
  structurally, so a pair boundary and a container boundary become the same
  thing (`sep=','` turns `x=[1,2]` into three tokens);
* a key holding `=`, a space or `sep` -- keys are written unquoted, so
  `{'a=b': 1}` renders `a=b=1` and reads back as `{'a': 'b=1'}`;
* a `str` value holding `sep` -- `{'a': 'x;y'}` renders `a='x;y'` and the
  reader splits the string in half.

And one about text that fails loudly, but in the wrong process: a non-finite
float renders as the bare identifier `nan`/`inf`, which only `literal_eval`
finds out about, at read time, on the far side of the wire.
"""

import ast
import math

# Characters a separator may not be, because `render_pon` writes them itself
# and `parse_pon` reads them back structurally. `=` splits a pair; `,` `[` `]`
# `(` `)` delimit the containers `_render_value` emits with a bare `,`; `'`
# quotes a string; `\` cannot appear in a value at all. A `sep` drawn from this
# set makes the two functions disagree about where a token ends -- `sep=','` on
# `{'x': [1, 2]}` renders `x=[1,2]` and parses back as three tokens, the first
# of which is `x=[1` (`SyntaxError: '[' was never closed`). Separator
# configurability is this module's reason to exist, so the separators that do
# not work are rejected rather than left to fail downstream.
_SEP_FORBIDDEN = frozenset(",[]()='\"\\")


def _check_sep(sep):
    """One character, and not one of the grammar's own."""
    if not isinstance(sep, str) or len(sep) != 1:
        raise ValueError(
            f'pon: sep must be exactly one character, got {sep!r}')
    if sep in _SEP_FORBIDDEN:
        raise ValueError(
            f'pon: sep={sep!r} is part of PON\'s own grammar '
            f'({"".join(sorted(_SEP_FORBIDDEN))}) and cannot separate pairs')


def parse_pon(line: str, sep: str = ';') -> dict:
    """Parse `"k1=v1<sep>k2=v2..."` into `{k1: v1, k2: v2, ...}`.

    Each value is decoded with `ast.literal_eval`, so it must be a Python
    literal (int, float, quoted string, tuple, list, `True`/`False`/`None`);
    anything else -- notably a bare identifier or a call -- raises rather
    than executing.
    """
    _check_sep(sep)
    d = {}
    for assignment in filter(None, (a.strip() for a in line.split(sep))):
        if '=' not in assignment:
            # Naming the token AND the line: this is the hardened replacement
            # for parse_python's eval(), so it reads worker wire text that a
            # human never typed, and a bare `ValueError: not enough values to
            # unpack` says nothing about which field arrived malformed.
            raise ValueError(
                f'parse_pon: {assignment!r} is not a key=value pair (no '
                f'{"="!r}) in {line!r} (sep={sep!r})')
        k, v = assignment.split('=', maxsplit=1)
        d[k.strip()] = ast.literal_eval(v.strip())
    return d


def _check_key(k, sep):
    """A key must survive the same round trip its value does.

    `_render_value` guards values only, and a key is written to the wire
    unquoted and unescaped -- so `{'a=b': 1}` renders `a=b=1`, which
    `parse_pon` reads back as `{'a': 'b=1'}`: no error, wrong dict. A key with
    a space used to be caught by `render_pon`'s trailing `assert`, which
    vanishes under `python -O`, i.e. exactly the mode where corrupt wire text
    is hardest to trace. These are `ValueError`s for that reason.
    """
    if not isinstance(k, str) or not k:
        raise ValueError(
            f'render_pon: key {k!r} must be a non-empty str')
    if '=' in k:
        raise ValueError(
            f'render_pon: key {k!r} contains {"="!r}, which parse_pon reads as '
            f'the key/value boundary -- the pair would read back with a '
            f'truncated key and the rest of the key glued onto the value')
    if sep in k or ' ' in k:
        raise ValueError(
            f'render_pon: key {k!r} contains a separator (sep={sep!r}) or a '
            f'space; either one re-tokenizes the pair')
    if not k.isprintable():
        raise ValueError(
            f'render_pon: key {k!r} contains a non-printable character')


def _render_value(k, v, sep) -> str:
    """One PON value: `repr()`-equivalent, but with no spaces anywhere.

    Recurses through tuples/lists rather than calling `repr()` on them, because
    `repr` separates container elements with `', '` and a space breaks the wire
    protocol (see the module docstring). The single-element tuple keeps its
    trailing comma -- `(1,)`, not `(1)`, which would read back as a plain int.
    """
    if isinstance(v, str):
        if repr(v) != "'" + v + "'":
            raise ValueError(
                f"render_pon: value of {k!r} ({v!r}) does not round-trip "
                f"through a plain single-quoted repr() (it contains a quote, "
                f"backslash, or non-printable character) -- PON cannot "
                f"represent it")
        if ' ' in v:
            raise ValueError(
                f"render_pon: value of {k!r} ({v!r}) contains a space. A PON "
                f"string is passed as one token on the exaid/testrun wire, "
                f"which splits on ' ', so an embedded space silently "
                f"re-tokenizes the whole command")
        # The SEPARATOR, not just a space. These are two different hazards and
        # the space guard above covers only one of them: `{'a': 'x;y'}` at the
        # default sep renders `a='x;y'`, and parse_pon splits it mid-string
        # into `a='x` and `y'` -- a SyntaxError from literal_eval, or worse a
        # value that happens to parse. A str is the only value kind that can
        # carry an arbitrary character, so this is the only place it can hide.
        if sep in v:
            raise ValueError(
                f"render_pon: value of {k!r} ({v!r}) contains the separator "
                f"{sep!r}; parse_pon would split the string in half")
        return repr(v)
    if isinstance(v, float) and not math.isfinite(v):
        # repr(nan) is 'nan' -- a bare identifier, so it clears the space guard
        # below and then dies in parse_pon's ast.literal_eval, at read time, in
        # another process. Reachable: tune/utils.py already carries a
        # sanitize_float for the NaN/Inf that turn up in tuning data.
        raise ValueError(
            f'render_pon: value of {k!r} is {v!r}, which renders as a bare '
            f'identifier that ast.literal_eval (hence parse_pon) rejects; PON '
            f'cannot represent a non-finite float')
    if isinstance(v, tuple):
        inner = ','.join(_render_value(k, x, sep) for x in v)
        return f'({inner},)' if len(v) == 1 else f'({inner})'
    if isinstance(v, list):
        return '[' + ','.join(_render_value(k, x, sep) for x in v) + ']'
    out = repr(v)
    if ' ' in out:
        raise ValueError(
            f"render_pon: value of {k!r} ({v!r}) renders as {out!r}, which "
            f"contains a space; PON values must be space-free (see the module "
            f"docstring on the exaid/testrun wire protocol)")
    return out


def render_pon(d: dict, sep: str = ';') -> str:
    """Render `{k1: v1, k2: v2, ...}` as `"k1=v1<sep>k2=v2..."`.

    `parse_pon(render_pon(d)) == d` for any `d` this function accepts, and the
    result never contains a space. A `str` value must round-trip through a
    plain single-quoted `repr()` -- `repr(v) == "'" + v + "'"` -- or this
    raises `ValueError` naming the offending key; that precondition is what
    lets a reader outside Python strip exactly one pair of surrounding single
    quotes instead of implementing a full Python-repr unescaper.

    `sep`, the keys and the values are all checked against the SAME grammar,
    because the round-trip promise above is only true if all three respect it:
    a separator drawn from the container syntax, a key holding an `=`, or a
    string value holding the separator each produce text that parses without
    error into the wrong dict.
    """
    _check_sep(sep)
    for k in d:
        _check_key(k, sep)
    text = sep.join(f'{k}={_render_value(k, v, sep)}' for k, v in d.items())
    # Belt and braces: _render_value rejects spaces per value, but a caller
    # passing sep=' ' would reintroduce them at the joins, and that is legal --
    # a space-separated --signature string is a command line, not a single
    # wire token.
    # Only assert the property the per-value checks are responsible for.
    assert sep == ' ' or ' ' not in text, (
        f'render_pon: emitted a space with sep={sep!r}: {text!r}')
    return text
