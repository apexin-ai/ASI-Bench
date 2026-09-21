#!/usr/bin/env python3
"""Apply reviewed patches to explicit, clean, pinned upstream checkouts.

No downloads, package installs, credentials, or CAD execution. Preflight all
three checkouts before applying any patch. Refuse already-patched/dirty trees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BUNDLE = Path(__file__).resolve().parent


def git(source, *args):
    return subprocess.run(['git', '-C', str(source), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def apply(root, bundle=BUNDLE):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    pending = []
    for entry in manifest['servers']:
        source = root.resolve() / entry['id']
        patch = bundle / entry['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != entry['patch_sha256']:
            raise ValueError(f"Patch checksum mismatch: {entry['id']}")
        # Don't accidentally apply relative to a parent Git repo.
        if Path(git(source, 'rev-parse', '--show-toplevel')).resolve() != source:
            raise ValueError(f'Not a checkout root: {source}')
        if git(source, 'rev-parse', 'HEAD') != entry['revision']:
            raise ValueError(f"Wrong upstream revision: {entry['id']}")
        git(source, 'diff', '--quiet')
        git(source, 'diff', '--cached', '--quiet')
        git(source, 'apply', '--check', str(patch.resolve()))
        pending.append((source, patch))
    for source, patch in pending:
        git(source, 'apply', str(patch.resolve()))
        print(f'Applied {patch.name} to {source}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    apply(args.source_root)
