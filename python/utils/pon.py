# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PON (Plain / Python Object Notation): a safe, separator-configurable
`k=v;k=v` wire format, and the one home for both reading and writing it.

The format's original reader was `aotriton.tune.utils.parse_python`, removed
here: it split a line on `;`, then on `=` with `maxsplit=1`, then called
`eval(v)` on the value. Two problems. The `eval` executed whatever the wire
carried -- and this wire carries text a worker process wrote, not text a human
typed. And the separator was hard-coded to `;`, so a caller whose wire
separator is something else (a space-separated `--signature`/`--hints` command
line, say) could not reuse it and wrote its own splitter instead.

`parse_pon` keeps the shape and fixes both: `ast.literal_eval` accepts exactly
the forms build inputs use -- ints, floats, quoted strings, tuples, lists,
`True`/`False`/`None` -- and raises on anything else, including bare
identifiers. `sep` defaults to `';'`, which is what made it a drop-in for the
two readers that used to call `eval` (`FlashEntry.parse_text`,
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

**EVERY `k=v` PAIR CONTAINS NO SPACES, whatever `sep` is.** That is a hard
property of the format, not a cosmetic one, and it is why `render_pon` cannot
simply call `repr()` on a container: `repr((1, 2))` is `'(1, 2)'`, with a space
after the comma.

Note what this does and does not say. The invariant is on the *pair*, not on
the rendered string: with the default `sep=';'` the whole string is space-free,
but with `sep=' '` the separators themselves are spaces and the string is a
command line by construction. Both are the same rule -- no space may appear
anywhere a reader would not expect a token boundary.

Two consumers pin it from opposite directions. The exaid/testrun wire protocol
needs the `;` form to survive as ONE token: `ExaidProxy.write`
(`python/tune/exaid.py`) joins its arguments with `' '` and the worker's
`first()` (`python/tune/testrun.py`) splits the line back apart on `' '`, so an
embedded space does not corrupt a value -- it silently re-tokenizes the whole
command, and a later positional argument is read from the middle of an earlier
one. The cmake/ninja rule generator needs the `' '` form for the opposite
reason: `;` is cmake's own list separator when it reads a file, so a
`;`-joined string would be split by cmake before the rule ever saw it. There
the space-free pair is what keeps each `k=v` a single cmake token.

**A value may therefore never contain a space, under any `sep`.** This is a
deliberate limitation and not an oversight: it keeps the strings short and the
grammar small enough that no consumer needs a quoting layer, and nothing in
this tree wants a space inside a value. A future caller that genuinely does
should encode it -- U+2423 OPEN BOX, a full-width space, or a private-use
codepoint -- rather than teaching every reader to unescape.

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

    Every malformed input raises `ValueError` naming the offending key or
    token and the line it came from, whatever `literal_eval` raised
    underneath; the original is chained as `__cause__`.
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
        k, v = (s.strip() for s in assignment.split('=', maxsplit=1))
        try:
            d[k] = ast.literal_eval(v)
        except (ValueError, TypeError, SyntaxError) as e:
            # Same reason as the branch above, and the same error type on
            # purpose. `literal_eval` reports SyntaxError for most malformed
            # wire text -- an unterminated string, a truncated list, a bad
            # numeric literal -- and ValueError only for text that parses but
            # is not a literal, such as a bare identifier. Letting both
            # propagate untouched leaves "PON could not read this" as two
            # different exception types, one of which names neither the key,
            # the token, nor the line it came from.
            #
            # Narrowing to ValueError makes the failure one catchable thing
            # and matches what this function already raises for a pair with
            # no `=`. `from e` keeps the original type and message reachable
            # as __cause__, so the distinction between "not a literal" and
            # "malformed syntax" is preserved for anyone who needs it -- it
            # simply stops being the caller's problem to discriminate.
            raise ValueError(
                f'parse_pon: value of {k!r} is not a Python literal: {v!r} '
                f'in {line!r} (sep={sep!r})') from e
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

    `parse_pon(render_pon(d)) == d` for any `d` this function accepts, and
    every `k=v` pair it emits is space-free. The rendered string as a whole is
    therefore space-free too under the default `sep=';'`, and contains spaces
    only at the joins when `sep=' '`. A `str` value must round-trip through a
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
    pairs = [f'{k}={_render_value(k, v, sep)}' for k, v in d.items()]
    # The invariant checked where it is stated: on the PAIR, not on the joined
    # string. Checking the join instead has to special-case sep=' ' -- and the
    # obvious spelling of that exemption, `sep == ' ' or ' ' not in text`,
    # disables the check in precisely the mode where a stray space is fatal
    # rather than merely untidy, since there a space IS the token boundary.
    #
    # `_check_key` and `_render_value` already reject a space on either side of
    # the `=`, so this cannot fire today. It is here so a rendering path added
    # around them later cannot quietly emit a pair that re-tokenizes -- and it
    # raises rather than asserts because `python -O` drops asserts, and a build
    # running under -O is exactly where corrupt wire text is hardest to trace.
    for pair in pairs:
        if ' ' in pair:
            raise ValueError(
                f'render_pon: pair {pair!r} contains a space; every k=v pair '
                f'must be space-free whatever sep is (sep={sep!r})')
    return sep.join(pairs)
