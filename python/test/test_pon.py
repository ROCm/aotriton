# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for python/utils/pon.py: `parse_pon` / `render_pon` round-trip
and the build-time rejection of un-representable strings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from aotriton.utils import parse_pon, render_pon


def test_round_trip_scalars():
    d = {'a': 1, 'b': -1, 'flag': True, 'off': False, 'nothing': None,
         'pi': 3.5, 'name': 'transposed'}
    assert parse_pon(render_pon(d)) == d


def test_round_trip_tuple_and_list():
    d = {'shape': (1, 2), 'items': [0, 1, 2], 'nested': (('a', 'b'), 3)}
    assert parse_pon(render_pon(d)) == d


def test_render_pon_emits_no_spaces():
    # THE wire-protocol invariant. ExaidProxy.write joins its arguments with
    # ' ' and testrun's first() splits them back on ' ', so one space inside a
    # PON token does not corrupt that value -- it re-tokenizes the whole
    # command. repr((1, 2)) would have produced '(1, 2)'.
    text = render_pon({'shape': (1, 2), 'items': [0, 1, 2],
                       'nested': (('a', 'b'), 3), 'name': 'transposed'})
    assert ' ' not in text, text


def test_render_pon_single_element_tuple_keeps_comma():
    # '(1)' would read back as the int 1, silently changing the type.
    assert render_pon({'t': (1,)}) == 't=(1,)'
    assert parse_pon(render_pon({'t': (1,)})) == {'t': (1,)}


def test_render_pon_rejects_string_with_space():
    with pytest.raises(ValueError, match='spacey'):
        render_pon({'spacey': 'two words'})


def test_round_trip_custom_separator():
    d = {'BLOCK_M': 64, 'CAUSAL': True}
    text = render_pon(d, sep=' ')
    assert ';' not in text
    assert parse_pon(text, sep=' ') == d


def test_every_pair_is_space_free_under_a_space_separator():
    # The invariant is on the PAIR, not on the joined string. With sep=' ' the
    # joins are spaces by construction, so "no spaces anywhere" is the wrong
    # assertion -- and the obvious way to write it, exempting sep=' ', disables
    # the check in the one mode where a stray space is fatal rather than untidy,
    # because here a space IS the token boundary.
    d = {'BLOCK_M': 64, 'name': 'transposed', 'shape': (1, 2)}
    text = render_pon(d, sep=' ')
    assert parse_pon(text, sep=' ') == d
    pairs = text.split(' ')
    assert len(pairs) == len(d), text      # no pair split itself in two
    for pair in pairs:
        assert '=' in pair and ' ' not in pair, pair


def test_render_pon_rejects_string_with_space_under_any_separator():
    # Not a property of the ';' wire alone. Under sep=' ' a space-bearing value
    # is not merely untransportable, it is unparseable: the reader splits mid
    # string and ast.literal_eval sees an unterminated literal. So the refusal
    # is unconditional, and a caller wanting a space in a value encodes it
    # (U+2423, a full-width space) rather than the format growing a quoting
    # layer every reader would have to implement.
    for sep in (';', ' ', '|'):
        with pytest.raises(ValueError, match='spacey'):
            render_pon({'spacey': 'two words'}, sep=sep)


def test_render_pon_quotes_strings():
    # The unified (quoted) dialect: a str value is rendered with its repr(),
    # not bare -- 'transposed', never transposed.
    assert render_pon({'v_lds_layout': 'transposed'}) == "v_lds_layout='transposed'"


def test_render_pon_rejects_unrepresentable_string():
    # A string whose repr() is not a plain single-quoted form (here: it
    # contains a single quote itself) cannot be represented by this grammar --
    # render_pon must raise, naming the offending key, rather than silently
    # emitting text that only a full Python-repr unescaper could read back.
    with pytest.raises(ValueError, match='bad_key'):
        render_pon({'bad_key': "can't"})


def test_render_pon_rejects_separator_inside_a_string():
    # NOT the same guard as the space one: at the default sep this renders
    # a='x;y', which parse_pon splits mid-string.
    with pytest.raises(ValueError, match='sep_in_value'):
        render_pon({'sep_in_value': 'x;y'})


def test_render_pon_rejects_a_separator_from_its_own_grammar():
    # sep=',' collides with the container separator _render_value emits, so
    # x=[1,2] would come back as three tokens, the first being 'x=[1'.
    with pytest.raises(ValueError, match='grammar'):
        render_pon({'x': [1, 2]}, sep=',')


def test_render_pon_rejects_a_key_with_an_equals_sign():
    # a=b=1 parses without error into {'a': 'b=1'} -- the wrong dict, quietly.
    with pytest.raises(ValueError, match='a=b'):
        render_pon({'a=b': 1})


def test_render_pon_rejects_a_key_with_a_space():
    # An assert would vanish under `python -O`, which is where corrupt wire
    # text is hardest to trace; this is a ValueError for that reason.
    with pytest.raises(ValueError, match='two words'):
        render_pon({'two words': 1})


def test_render_pon_rejects_non_finite_floats():
    # repr(nan) is the bare identifier 'nan': it clears the space guard and
    # then dies in the reader's ast.literal_eval, in another process.
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            render_pon({'nonfinite': bad})


def test_parse_pon_names_the_token_that_has_no_equals():
    with pytest.raises(ValueError, match='garbage'):
        parse_pon('a=1;garbage;b=2')


def test_flash_entry_round_trips_through_pon():
    from aotriton.tune.registry import load_flash_entry_module

    modules_dir = Path(__file__).resolve().parents[2] / 'modules'
    FlashEntry = load_flash_entry_module(modules_dir=modules_dir).FlashEntry

    e = FlashEntry(dtype='bfloat16', hdim=(64, 128), seqlen_q=256, seqlen_k=512,
                   causal=True, dropout_p=0.5, bias_type=1)
    d = parse_pon(e.as_text())
    assert FlashEntry(**d) == e


def main():
    """Standalone runner. Every test here runs without pytest -- `pytest.raises`
    is a plain context manager, not a fixture -- so nothing is skipped and a
    failure is a non-zero exit. Same shape as test_choices_view.py's."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f'FAIL: {fn.__name__}: {type(e).__name__}: {e}')
    if failures:
        print(f'FAILED: {failures} of {len(fns)} pon tests.')
        return 1
    print(f'OK: {len(fns)} pon tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
