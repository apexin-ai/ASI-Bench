"""Convert downloaded ASI-Bench seed31415 assets into a BenchFlow task package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchflow.task import render_task_md


@dataclass(frozen=True)
class PrepareConfig:
    raw_dir: Path
    task_id: str
    instance_id: str
    level: str
    output_dir: Path
    runtime_profile: str = 'cpu-v1'
    timeout_seconds: int = 3600
    verifier_timeout_seconds: int = 600
    converter_version: str = 'prepare-v5'

@dataclass(frozen=True)
class PreparedTask:
    output_dir: Path
    manifest_path: Path
    status: str
    task_id: str
    instance_id: str
    level: str

def _json(p): return json.loads(p.read_text())
def _yaml(p): return yaml.safe_load(p.read_text()) or {}
def _sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def _safe_rel(value):
    if not isinstance(value,str) or not value or value.startswith(('/', '\\')) or '\\' in value or any(part == '' for part in value.split('/')):
        raise ValueError(f'unsafe relative path: {value!r}')
    p=Path(value)
    if any(x in ('','.','..') for x in p.parts): raise ValueError(f'unsafe relative path: {value!r}')
    return p

def _ids(task_id, instance_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', task_id): raise ValueError('invalid task id')
    if instance_id != task_id + '__seed31415': raise ValueError('instance must be the seed31415 task instance')

def select_prompt(instance, level):
    if level not in {'b1','b2','b3','b4'}: raise ValueError('level must be b1, b2, b3, or b4')
    p=instance/f'prompt_{level}.md'
    if not p.is_file(): raise ValueError(f'missing {p.name}')
    return p

def resolve_input_files(instance, meta):
    result=[]; seen=set()
    for item in (meta.get('input') or {}).get('files') or []:
        name=_safe_rel(item.get('name'))
        rel=name if name.parts[0]=='data' else Path('data')/name
        if str(rel).casefold() in seen: raise ValueError(f'duplicate input: {rel}')
        seen.add(str(rel).casefold()); src=instance/rel
        if not src.is_file(): raise ValueError(f'missing input file: {rel}')
        result.append((src, rel))
    return result

def expected_outputs(meta):
    out=[]
    for item in (meta.get('output') or {}).get('files') or []:
        name=_safe_rel(item.get('name'))
        typ=item.get('type')
        if not isinstance(typ,str) or not typ: raise ValueError('output type is required')
        out.append({'name':str(name),'type':typ})
    return out

def anonymous_task_id(task_id): return 'task_' + hashlib.sha256(task_id.encode()).hexdigest()[:12]

def build_task_info(task_id, level, meta, timeout):
    return {'schema_version':3,'task_id':anonymous_task_id(task_id),'prompt_level':level,'expected_outputs':expected_outputs(meta),'timeout_seconds':timeout}

def build_task_md(prompt, task_id, level, timeout, verifier_timeout):
    front={'metadata':{'benchmark':'ASI-Bench','task_id':anonymous_task_id(task_id),'prompt_level':level,'official':False},'agent':{'timeout_sec':timeout},'verifier':{'timeout_sec':verifier_timeout},'sandbox':{'cpus':2,'memory_mb':4096,'workdir':'/workspace'}}
    return render_task_md(front, prompt)

def build_instance_parameters(meta): return dict(meta.get('params_used') or {})

def validate_runtime(meta, profile):
    if profile != 'cpu-v1': raise ValueError('only cpu-v1 is supported')
    runtime=meta.get('runtime') or {}
    if runtime.get('gpu') or runtime.get('network') is True: raise ValueError('task runtime is incompatible with cpu-v1')
    if runtime.get('python') and not str(runtime['python']).startswith('>=3.11'): raise ValueError('unsupported Python runtime')

def build_dockerfile(meta: dict[str, Any]) -> str:
    """Install public task dependencies before the network-isolated attempt.

    This minimal converter accepts index package/version specs only, not URLs,
    pip options, markers or arbitrary shell/Dockerfile fragments.
    """
    packages = (meta.get('runtime') or {}).get('packages', [])
    pattern = r'[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?(?:(?:===|==|~=|>=|<=|!=|>|<)[A-Za-z0-9.*+!_-]+(?:\.[A-Za-z0-9.*+!_-]+)*(?:,(?:===|==|~=|>=|<=|!=|>|<)[A-Za-z0-9.*+!_-]+)*)?'
    if not isinstance(packages, list) or any(
        not isinstance(p, str) or not re.fullmatch(pattern, p) for p in packages
    ):
        raise ValueError('unsupported runtime.packages; expected index package/version specs')
    # A single resolver sees both scoring and task requirements. Conflicting
    # specifications must fail the build, never silently override each other.
    packages = list(dict.fromkeys(['numpy', 'PyYAML', 'packaging', *packages]))
    install = 'RUN ' + json.dumps(['python', '-m', 'pip', 'install', '--no-cache-dir',
                                  '-c', '/opt/runtime-constraints.txt', *packages]) + '\n'
    return ('FROM python:3.11-slim\nCOPY runtime-constraints.txt /opt/runtime-constraints.txt\n'
            + install + 'WORKDIR /workspace\nCOPY inputs/ /workspace/\nCMD ["sleep", "infinity"]\n')


def _copy(src,dst): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def _files(root): return [p for p in root.rglob('*') if p.is_file()]

class InventoryError(ValueError):
    """Safe CLI diagnostic containing only relative inventory paths."""


def is_inventory_noise(path: Path) -> bool:
    """Only ordinary Finder metadata is noise; never hide links/special files."""
    return path.name == '.DS_Store' and stat.S_ISREG(path.lstat().st_mode)


def check_inventory(root: Path, actual: set[str], expected: set[str], label: str) -> set[str]:
    actual = {name for name in actual
              if name in expected or not is_inventory_noise(root / name)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        # JSON escapes unusual filenames and bounds diagnostic volume.
        detail = {key: {'count': len(paths), 'paths': [p[:240] for p in paths[:10]]}
                  for key, paths in [('missing', missing), ('extra', extra)] if paths}
        raise InventoryError(f'{label}: ' + json.dumps(detail, ensure_ascii=True))
    return actual


def validate_source_files(root: Path, manifest: dict[str, Any]) -> None:
    """Check the complete downloaded inventory; never follow source symlinks."""
    indexed = {}
    for entry in manifest.get('files', []):
        relative = _safe_rel(entry['path']).as_posix()
        if relative in indexed or relative == 'sources.json':
            raise ValueError('invalid raw source inventory')
        indexed[relative] = entry['sha256']
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('unsafe raw source file')
        if path.is_file() and path != root/'sources.json':
            actual.add(path.relative_to(root).as_posix())
    if not indexed:
        raise InventoryError('raw source inventory is empty')
    check_inventory(root, actual, set(indexed), 'raw source integrity inventory mismatch')
    for relative, digest in indexed.items():
        if _sha(root/relative) != digest:
            raise InventoryError('raw source integrity mismatch: ' + json.dumps(relative))


def resolve_source_assets(manifest: dict[str, Any]) -> Path:
    """Resolve prepare-v3's read-only cache dependency and verify its binding."""
    assets = manifest.get('source_assets', {})
    value = assets.get('raw_dir')
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('missing absolute source_assets.raw_dir; regenerate prepared task')
    root = Path(value)
    source_path = root/'sources.json'
    if root.is_symlink() or source_path.is_symlink() or not source_path.is_file():
        raise ValueError('source assets missing or unsafe')
    if _sha(source_path) != assets.get('manifest_sha256'):
        raise ValueError('source manifest integrity mismatch')
    source = _json(source_path)
    if (source.get('seed') != 31415
            or source.get('task_id') != manifest['task']['id']
            or source.get('instance_id') != manifest['instance']['id']
            or {k: source.get(k, {}) for k in ('hf', 'github')} != manifest['sources']):
        raise ValueError('source assets identity mismatch')
    validate_source_files(root, source)
    reference = root/'instance'/manifest['instance']['id']/'reference'
    if not reference.is_dir() or not any(p.is_file() for p in reference.rglob('*')):
        raise ValueError('source reference missing or empty')
    return root.resolve()


def prepare_task(c: PrepareConfig) -> PreparedTask:
    from benchmarks.asi_bench.scorer import (
        ADAPTER_ROOT,
        load_evaluator_spec,
        scoring_profile_identity,
    )

    if c.converter_version != 'prepare-v5':
        raise ValueError('only prepare-v5 can be generated; regenerate prepared assets')
    spec = load_evaluator_spec()
    identity = scoring_profile_identity(spec)
    # Read once and verify the exact bytes that will be copied and keyed.
    templates = {}
    for field, name in (('launcher', 'test.sh'), ('bridge', 'score_entry.py'), ('scoring', 'scoring.py')):
        data = (ADAPTER_ROOT / spec[field]['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec[field]['sha256']:
            raise ValueError(f'{field} template digest mismatch')
        templates[name] = data
    task_dir=c.raw_dir/'task_bundle'/'tasks'/Path(*c.task_id.split('.'))
    instance=c.raw_dir/'instance'/c.instance_id
    _ids(c.task_id,c.instance_id)
    if not instance.is_dir() or not task_dir.is_dir(): raise ValueError('raw instance or task bundle is missing')
    meta=_yaml(task_dir/'task_meta.yaml'); evaluation=_yaml(task_dir/'task_eval.yaml')
    if meta.get('id') != c.task_id or evaluation.get('task_id') not in (None,c.task_id): raise ValueError('task metadata identity mismatch')
    source_manifest = _json(c.raw_dir/'sources.json')
    if source_manifest.get('seed') != 31415 or source_manifest.get('task_id') != c.task_id or source_manifest.get('instance_id') != c.instance_id:
        raise ValueError('raw source manifest identity mismatch')
    validate_source_files(c.raw_dir, source_manifest)
    if not (instance/'reference').is_dir() or not any((p.is_file() and not p.is_symlink()) for p in (instance/'reference').rglob('*')):
        raise ValueError('instance reference is missing or empty')
    validate_runtime(meta,c.runtime_profile); prompt=select_prompt(instance,c.level); inputs=resolve_input_files(instance,meta)
    info=build_task_info(c.task_id,c.level,meta,c.timeout_seconds)
    params=build_instance_parameters(_json(instance/'instance_meta.json'))
    hf_sha = source_manifest.get('hf', {}).get('resolved_revision', '')
    asi_sha = source_manifest.get('github', {}).get('resolved_revision', '')
    scoring = identity
    scoring_key = json.dumps({'constraints': _sha(Path(__file__).with_name('runtime-constraints.txt')), 'profile': scoring, 'identity': identity}, sort_keys=True, separators=(',', ':'))
    key=hashlib.sha256(f'{c.task_id}|{c.instance_id}|{c.level}|{c.converter_version}|{c.runtime_profile}|{hf_sha}|{asi_sha}|{c.timeout_seconds}|{c.verifier_timeout_seconds}|{scoring_key}'.encode()).hexdigest()[:16]
    out=c.output_dir/c.task_id/key
    if out.is_symlink(): raise ValueError(f'unsafe prepared output: {out}')
    c.output_dir.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.prepare-',dir=c.output_dir))
    try:
        _copy(prompt,staging/'environment/inputs/prompt.md')
        (staging/'environment/inputs/task_info.json').parent.mkdir(parents=True,exist_ok=True); (staging/'environment/inputs/task_info.json').write_text(json.dumps(info,indent=2)+'\n')
        for src,rel in inputs: _copy(src,staging/'environment/inputs'/rel)
        (staging/'verifier').mkdir()
        for name, data in templates.items():
            (staging/'verifier'/name).write_bytes(data)
        (staging/'verifier/instance_parameters.json').write_text(json.dumps(params,indent=2)+'\n')
        profile={'profile_id':'cpu-v1','variant':'os','cpus':2,'memory_mb':4096,'workdir':'/workspace','network':False,'base_image':'python:3.11-slim'}
        (staging/'environment/runtime-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        _copy(Path(__file__).with_name('runtime-constraints.txt'), staging/'environment/runtime-constraints.txt')
        (staging/'environment/Dockerfile').write_text(build_dockerfile(meta))
        (staging/'task.md').write_text(build_task_md(prompt.read_text(),c.task_id,c.level,c.timeout_seconds,c.verifier_timeout_seconds))
        files=[]
        for p in _files(staging):
            files.append({'target':str(p.relative_to(staging)),'size':p.stat().st_size,'sha256':_sha(p),'visibility':['agent'] if str(p.relative_to(staging)).startswith('environment/inputs/') else ['scoring']})
        manifest={'schema_version':1,'converter_version':c.converter_version,'status':'prepared','benchmark':'ASI-Bench','official':False,'seed':31415,'task':{'id':c.task_id,'anonymous_id':anonymous_task_id(c.task_id),'level':c.level,'timeout_seconds':c.timeout_seconds},'instance':{'id':c.instance_id},'runtime':{'profile_id':'cpu-v1','path':'environment/runtime-profile.json'},'paths':{'task_document':'task.md','agent_inputs':'environment/inputs','instance_parameters':'verifier/instance_parameters.json'},'sources': {'hf': source_manifest.get('hf', {}), 'github': source_manifest.get('github', {})}, 'files':files}
        manifest['scoring'] = scoring
        manifest['source_assets'] = {'raw_dir': str(c.raw_dir.resolve()), 'manifest_sha256': _sha(c.raw_dir/'sources.json')}
        (staging/'task_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if out.exists():
            expected = {str(p.relative_to(staging)): _sha(p) for p in _files(staging)}
            if not out.is_dir() or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in out.rglob('*')):
                raise ValueError(f'unsafe prepared output: {out}')
            actual = {str(p.relative_to(out)): _sha(p) for p in _files(out)}
            names = check_inventory(out, set(actual), set(expected), 'existing prepared inventory mismatch')
            for name in sorted(names):
                if actual[name] != expected[name]:
                    raise InventoryError('existing prepared integrity mismatch: ' + json.dumps(name))
            shutil.rmtree(staging)
        else:
            out.parent.mkdir(parents=True,exist_ok=True); staging.rename(out)
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise
    return PreparedTask(out,out/'task_manifest.json','prepared',c.task_id,c.instance_id,c.level)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',type=Path,required=True); p.add_argument('--task-id',required=True); p.add_argument('--instance-id'); p.add_argument('--level',default='b1'); p.add_argument('--output-dir',type=Path,default=Path('prepared')); a=p.parse_args(); iid=a.instance_id or a.task_id+'__seed31415'; r=prepare_task(PrepareConfig(a.raw_dir,a.task_id,iid,a.level,a.output_dir)); print(json.dumps({'status':r.status,'output_dir':str(r.output_dir),'manifest':str(r.manifest_path)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
