"""Download raw seed31415 assets; no conversion, execution, or scoring."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx
import yaml
from huggingface_hub import HfApi, RepoFile, hf_hub_download

HF_REPO = 'Apexintelligence-AI/ASI-Bench-seed31415'
GITHUB_REPO = 'apexin-ai/ASI-Bench'
MAX_FILES = 10_000
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_CACHE_ROOT = (
    Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser()
    / 'benchflow' / 'asi_bench'
)
DEFAULT_HF_CACHE_DIR = DEFAULT_CACHE_ROOT / 'hf'
DEFAULT_DOWNLOADS_DIR = DEFAULT_CACHE_ROOT / 'downloads'


def validate_task_id(task_id: str) -> tuple[str, str]:
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', task_id):
        raise ValueError('task ID must be <domain>.<name> with no path separators')
    domain, name = task_id.split('.')
    return domain, name


def safe_path(value: str) -> Path:
    parts = value.split('/')
    if any(p in ('', '.', '..') for p in parts) or '\\' in value or '\x00' in value:
        raise ValueError(f'unsafe source path: {value!r}')
    return Path(*parts)


def checked_sha(value: str) -> str:
    if not re.fullmatch(r'[0-9a-f]{40}', value):
        raise ValueError('source did not resolve to an immutable Git commit')
    return value


class Budget:
    def __init__(self):
        self.total = 0
        self.paths: set[str] = set()

    def add(self, path: str, size: int):
        safe_path(path)
        key = path.casefold()
        if key in self.paths:
            raise ValueError(f'duplicate or case-colliding path: {path}')
        if size < 0 or size > MAX_FILE_BYTES:
            raise ValueError(f'file size exceeds acquisition limit: {path}')
        self.paths.add(key)
        self.total += size
        if len(self.paths) > MAX_FILES or self.total > MAX_TOTAL_BYTES:
            raise ValueError('asset acquisition limit exceeded')


def download_instance(task_id: str, destination: Path, revision: str,
                      cache_dir: Path | None) -> str:
    """Download exactly one published instance, including all prompts/reference."""
    token = os.environ.get('HF_TOKEN') or False
    api = HfApi(token=token)
    sha = checked_sha(api.dataset_info(HF_REPO, revision=revision).sha)
    prefix = f'tasks/{task_id}__seed31415'
    budget = Budget()
    for entry in api.list_repo_tree(
        HF_REPO, path_in_repo=prefix, recursive=True,
        repo_type='dataset', revision=sha,
    ):
        if not isinstance(entry, RepoFile):
            continue
        relative = str(PurePosixPath(entry.path).relative_to(prefix))
        budget.add(relative, entry.size)
        cached = Path(hf_hub_download(
            HF_REPO, entry.path, repo_type='dataset', revision=sha,
            token=token, cache_dir=cache_dir,
        ))
        if not cached.is_file():
            raise ValueError(f'HF downloaded file missing or invalid: {relative}')
        actual_size = cached.stat().st_size
        if actual_size != entry.size:
            raise ValueError(
                f'HF file size mismatch: {relative}; '
                f'expected={entry.size}, actual={actual_size}'
            )
        target = destination / safe_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Official HF cache uses links to blobs; materialize ordinary files.
        shutil.copyfile(cached, target)
    prompt_count = sum(p.is_file() for p in destination.glob('prompt_b[1-4].md'))
    if prompt_count != 4:
        raise ValueError(
            f'seed31415 instance prompt count mismatch: expected=4, actual={prompt_count}'
        )
    reference = destination / 'reference'
    if not reference.is_dir() or not any(p.is_file() for p in reference.rglob('*')):
        raise ValueError('seed31415 instance is missing populated reference/')
    return sha


def download_task_bundle(task_id: str, destination: Path, revision: str) -> str:
    """Walk a pinned Git subtree and download every regular blob, no allowlist."""
    domain, name = validate_task_id(task_id)
    headers = {'Accept': 'application/vnd.github+json'}
    if token := os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f'Bearer {token}'
    with httpx.Client(
        base_url=f'https://api.github.com/repos/{GITHUB_REPO}/',
        headers=headers, timeout=60,
    ) as client:
        def get(path):
            for attempt in range(3):
                try:
                    response = client.get(path)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as error:
                    if attempt == 2 or error.response.status_code not in (429, 502, 503, 504):
                        raise
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                time.sleep(2 ** attempt)
            raise AssertionError("unreachable")

        def tree(sha):
            payload = get(f'git/trees/{sha}')
            if payload.get('truncated'):
                raise ValueError('GitHub returned an incomplete tree')
            return payload['tree']

        commit = get(f'commits/{quote(revision, safe="")}')
        sha = checked_sha(commit['sha'])
        tree_sha = commit['commit']['tree']['sha']
        # Traverse by tree SHA instead of enumerating/downloading the whole repo.
        for part in ('tasks', domain, name):
            matches = [e for e in tree(tree_sha) if e['path'] == part]
            if len(matches) != 1 or matches[0]['type'] != 'tree':
                raise ValueError(f'task folder not found: {task_id}')
            tree_sha = matches[0]['sha']

        budget = Budget()
        pending = [('', tree_sha)]
        directories = 0
        while pending:
            prefix, current_sha = pending.pop()
            directories += 1
            if directories > MAX_FILES:
                raise ValueError('too many task subdirectories')
            for entry in tree(current_sha):
                relative = prefix + entry['path']
                path = safe_path(relative)
                if entry['type'] == 'tree':
                    if len(path.parts) > 64:
                        raise ValueError('task directory nesting limit exceeded')
                    pending.append((relative + '/', entry['sha']))
                    continue
                if entry['type'] != 'blob' or entry['mode'] not in ('100644', '100755'):
                    raise ValueError(f'unsupported Git entry: {relative}')
                budget.add(relative, entry['size'])
                blob = get(f'git/blobs/{entry["sha"]}')
                if blob['encoding'] != 'base64':
                    raise ValueError(f'unsupported blob encoding: {relative}')
                data = base64.b64decode(''.join(blob['content'].split()), validate=True)
                digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                if len(data) != entry['size'] or digest != entry['sha']:
                    raise ValueError(f'Git blob integrity mismatch: {relative}')
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if entry["mode"] == "100755" else 0o644)
    return sha


def acquire(task_id: str, output_dir: Path | None = None, *, hf_revision: str = 'main',
            asi_revision: str = 'main', cache_dir: Path | None = None) -> dict:
    domain, name = validate_task_id(task_id)
    if output_dir is None:
        output_dir = DEFAULT_DOWNLOADS_DIR / task_id
    if cache_dir is None:
        cache_dir = DEFAULT_HF_CACHE_DIR
    output_dir = Path(output_dir).expanduser().absolute()
    cache_dir = Path(cache_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'refusing to overwrite {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents two cooperating invocations from publishing the same output.
    lock = output_dir.with_name(output_dir.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f'refusing to overwrite {output_dir}')
        with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}-',
                                         dir=output_dir.parent) as temporary:
            staging = Path(temporary)
            instance_id = f'{task_id}__seed31415'
            instance = Path('instance') / instance_id
            bundle = Path('task_bundle/tasks') / domain / name
            hf_sha = download_instance(task_id, staging / instance, hf_revision, cache_dir)
            git_sha = download_task_bundle(task_id, staging / bundle, asi_revision)
            for filename, field in [('task_meta.yaml', 'id'), ('task_eval.yaml', 'task_id')]:
                metadata = yaml.safe_load((staging / bundle / filename).read_text())
                if not isinstance(metadata, dict) or metadata.get(field) != task_id:
                    raise ValueError(f'task identity mismatch in {filename}')
            files = []
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    with path.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    files.append({'path': path.relative_to(staging).as_posix(),
                                  'size': path.stat().st_size, 'sha256': digest})
            manifest = {
                'schema_version': 1, 'status': 'downloaded', 'seed': 31415,
                'task_id': task_id, 'instance_id': instance_id,
                'instance_dir': instance.as_posix(), 'task_bundle_dir': bundle.as_posix(),
                'hf': {'repo': HF_REPO, 'requested_revision': hf_revision,
                       'resolved_revision': hf_sha, 'path': f'tasks/{instance_id}'},
                'github': {'repo': GITHUB_REPO, 'requested_revision': asi_revision,
                           'resolved_revision': git_sha, 'path': f'tasks/{domain}/{name}'},
                'files': files,
            }
            (staging / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
            staging.rename(output_dir)
            return manifest
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--hf-revision', default='main')
    parser.add_argument('--asi-revision', default='main')
    args = parser.parse_args()
    manifest = acquire(**vars(args))
    print(json.dumps({key: manifest[key] for key in (
        'status', 'instance_dir', 'task_bundle_dir', 'hf', 'github'
    )}, indent=2))


if __name__ == '__main__':
    main()
