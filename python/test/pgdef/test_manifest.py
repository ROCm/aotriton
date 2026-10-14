# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Application manifests: the lookup, tune's manifest, and that every key the
`.tune` scripts ask for exists."""

import pathlib
import re
import shutil
import subprocess

import pytest

from aotriton.pgdef import manifest as pgmanifest

_REPO = pathlib.Path(__file__).resolve().parents[3]

SAMPLE = {
    'app': 'x',
    'image': {'dockerfile': 'image.build/Dockerfile', 'tag_suffix': '-x_{arch}'},
    'env': {'kernel': ['A=/{arch}'], 'op': ['A=/t/{arch}', 'B=1'], '*': ['C=2']},
    'paths': ['a', 'b'],
    'flag': True,
}


def test_lookup_nested_keys_and_placeholders():
    assert pgmanifest.lookup(SAMPLE, 'app') == 'x'
    assert pgmanifest.lookup(SAMPLE, 'image.tag_suffix', arch='gfx950') == '-x_gfx950'
    assert pgmanifest.lookup(SAMPLE, 'paths') == ['a', 'b']
    assert pgmanifest.lookup(SAMPLE, 'flag') is True


def test_lookup_selects_by_category_level_with_a_wildcard_fallback():
    assert pgmanifest.lookup(SAMPLE, 'env', cat_level1='kernel', arch='g') == ['A=/g']
    assert pgmanifest.lookup(SAMPLE, 'env', cat_level1='op', arch='g') == ['A=/t/g', 'B=1']
    assert pgmanifest.lookup(SAMPLE, 'env', cat_level1='other') == ['C=2']


def test_lookup_missing_key():
    for key in ('nope', 'image.nope', 'app.deeper'):
        with pytest.raises(KeyError):
            pgmanifest.lookup(SAMPLE, key)


def test_render():
    assert pgmanifest.render('s') == 's'
    assert pgmanifest.render(['a', 'b']) == 'a\nb'
    assert pgmanifest.render([]) == ''
    assert pgmanifest.render({'a': [1]}) == '{"a": [1]}'
    assert pgmanifest.render(True) == 'true'


def test_main_prints_and_reports_a_missing_key(capsys):
    assert pgmanifest.main(SAMPLE, ['image.tag_suffix', '--arch', 'gfx942']) == 0
    assert capsys.readouterr().out == '-x_gfx942\n'
    assert pgmanifest.main(SAMPLE, ['nope']) == 1
    assert 'no key' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# tune's manifest
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tune_manifest():
    # aotriton.tune is its own distribution (aotriton-tune); this suite may run
    # without it installed.
    return pytest.importorskip('aotriton.tune.manifest').MANIFEST


def test_tune_manifest_is_complete_for_every_level(tune_manifest):
    assert tune_manifest['app'] == 'tune'
    levels = tune_manifest['category']['level1_values']
    assert levels == ['kernel', 'op']
    for level in levels:
        for key in ('install.subtree', 'fetchbuild.path', 'fetchbuild.exclude'):
            pgmanifest.lookup(tune_manifest, key, cat_level1=level, arch='gfx950')
    assert pgmanifest.lookup(tune_manifest, 'install.subtree', cat_level1='op', arch='gfx950') \
        == 'installed/test/gfx950'


def test_tune_manifest_deploy_role(tune_manifest):
    assert tune_manifest['deploy']['role'] == 'Tuner'


def _script_manifest_keys():
    """(file, key) for every `app_manifest <app> <key>` in the .tune scripts."""
    pat = re.compile(r'app_manifest\s+"?\$?\{?\w+\}?"?\s+([a-z0-9_]+(?:\.[a-z0-9_]+)*)')
    for path in sorted((_REPO / '.tune').rglob('*')):
        if not path.is_file() or path.suffix in {'.html', '.js', '.css', '.png', '.md'} \
                or 'pkgroot' in path.parts or path.name == 'manifest.sh':
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for key in pat.findall(text):
            yield str(path.relative_to(_REPO)), key


@pytest.mark.parametrize('app,cat_level1', [('tune', 'kernel'), ('tune', 'op')])
def test_every_key_a_script_reads_exists_in_every_applications_manifest(app, cat_level1):
    """The scripts are generic: a key one of them asks for must exist for every
    registered application, or that application's script run dies on a missing key."""
    manifest = pytest.importorskip(f'aotriton.{app}.manifest').MANIFEST
    keys = list(_script_manifest_keys())
    assert keys, 'the scripts do not read the manifest at all?'
    missing = []
    for path, key in keys:
        try:
            pgmanifest.lookup(manifest, key, cat_level1=cat_level1, arch='gfx950')
        except KeyError:
            missing.append((path, key))
    assert not missing, missing


@pytest.mark.skipif(shutil.which('bash') is None or shutil.which('python3') is None, reason='needs bash and python3')
def test_the_shell_helper_works_from_a_bare_checkout():
    # Uses .tune/lib/pkgroot/aotriton -> python/, so no install is needed (and it
    # does not matter whether aotriton.tune is installed here).
    script = (f'cd {_REPO} && . .tune/lib/manifest.sh && require_app tune && '
              'app_manifest tune install.subtree --arch gfx942 --cat_level1 op')
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'installed/test/gfx942'
    bad = subprocess.run(['bash', '-c', f'cd {_REPO} && . .tune/lib/manifest.sh && require_app nope'],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and 'not registered' in bad.stderr
