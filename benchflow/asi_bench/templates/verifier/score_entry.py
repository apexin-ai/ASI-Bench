"""Container-only ASI scoring bridge: input binding, scoring and reports.

Scoring aggregation lives in the adapter-owned scoring module. Task-source
selection is copied from ASI-Bench @ 5935b5f33549e348a8505bb355d3b6f4fe4a273c.
Metadata is projected to scoring-only fields. The restricted evaluator package
must be supplied explicitly; never import the full orchestrator or a host copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import yaml
from ai4sci_bench.core.logger import get_logger
from scoring import _evaluate_gates_and_scores

logger = get_logger(__name__)

LEGACY_TASK_FILE = "task.yaml"
META_TASK_FILE = "task_meta.yaml"
EVAL_TASK_FILE = "task_eval.yaml"



def resolve_task_sources(task_dir: Path) -> tuple[Path, Path | None]:
    """Resolve the metadata source file(s) for a task directory.

    Supports two on-disk layouts:

    * **Legacy** — a single monolithic ``task.yaml`` → ``(task.yaml, None)``.
    * **Split** — ``task_meta.yaml`` (public) plus an optional
      ``task_eval.yaml`` (scorer-internal) → ``(task_meta.yaml, task_eval.yaml | None)``.
      A minimal/HF tree may ship only ``task_meta.yaml``; the public GitHub
      catalog also ships scoring-only ``task_eval.yaml`` files.

    When both ``task.yaml`` and ``task_meta.yaml`` exist (e.g. mid-migration),
    the legacy ``task.yaml`` wins so behaviour is unchanged until a task is
    fully migrated (its ``task.yaml`` removed).

    Raises:
        FileNotFoundError: if neither layout is present.
    """
    legacy = task_dir / LEGACY_TASK_FILE
    if legacy.exists():
        meta = task_dir / META_TASK_FILE
        if meta.exists():
            logger.warning(
                "Both %s and %s present in %s; using legacy %s",
                LEGACY_TASK_FILE, META_TASK_FILE, task_dir, LEGACY_TASK_FILE,
            )
        return legacy, None
    meta = task_dir / META_TASK_FILE
    if meta.exists():
        ev = task_dir / EVAL_TASK_FILE
        return meta, (ev if ev.exists() else None)
    raise FileNotFoundError(
        f"No {LEGACY_TASK_FILE} or {META_TASK_FILE} found in {task_dir}"
    )


def load_scoring_metadata(task_dir: Path) -> dict[str, Any]:
    """Load only id/runtime/evaluation, preserving native source selection.

    Caller owns input identity/path validation. This deliberately omits the
    original loader's generation parsing, runtime validation and private keys.
    It does not load a generator, task scorer, or prediction module.
    """
    meta_path, eval_path = resolve_task_sources(task_dir)
    with open(meta_path, encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if eval_path is not None:
        with open(eval_path, encoding="utf-8") as stream:
            eval_data = yaml.safe_load(stream)
        if isinstance(eval_data, dict):
            metadata.update({key: value for key, value in eval_data.items()
                             if key in {"id", "runtime", "evaluation"}})
    return {key: value for key, value in metadata.items()
            if key in {"id", "runtime", "evaluation"}}


# Only modules shipped in the verified evaluator tree may be imported.
# The orchestrator/generation stack is deliberately outside this bridge.
EXCLUDED_MODULES = (
    "ai4sci_bench.generators", "ai4sci_bench.adapters",
    "ai4sci_bench.runner.orchestrator",
)


def check_evaluator_imports(evaluator_root: Path) -> dict[str, str]:
    """Reject a full/foreign framework, including a second installed copy."""
    import sys

    root = evaluator_root.resolve()
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "ai4sci_bench" and not name.startswith("ai4sci_bench."):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in EXCLUDED_MODULES):
            raise ValueError(f"Unsupported evaluator module loaded: {name}")
        location = getattr(module, "__file__", None)
        if not location:
            raise ValueError(f"Evaluator module has no source file: {name}")
        actual = Path(location).resolve()
        relative = Path(*name.split("."))
        expected = root / relative
        if actual not in (expected.with_suffix(".py"), expected / "__init__.py"):
            raise ValueError(f"Evaluator import origin mismatch: {name}: {actual}")
        origins[name] = str(actual)
    return origins


def register_task_scorers(task_dir: Path, task_id: str, evaluation: dict[str, Any],
                          evaluator_root: Path) -> dict[str, Any]:
    """Register a verified task inside the evaluator container; never on host.

    The caller must validate task bytes/identity before loading this executable
    input. No score(), ensure_env(), dependency installation or prediction import.
    """
    import importlib
    import sys

    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    requested = set()
    for key in ("gates", "scoring"):
        configs = evaluation.get(key, [])
        if not isinstance(configs, list):
            raise ValueError(f"evaluation.{key} must be a list")
        for config in configs:
            if (not isinstance(config, dict) or not isinstance(config.get("scorer"), str)
                    or not config["scorer"].strip()):
                raise ValueError(f"Invalid scorer config in {key}: {config}")
            requested.add(config["scorer"])
    check_evaluator_imports(evaluator_root)
    registry = importlib.import_module("ai4sci_bench.core.scorer")
    for module_path in sorted((evaluator_root / "ai4sci_bench/scorers").glob("*.py")):
        if not module_path.stem.startswith("_") and module_path.stem != "custom":
            importlib.import_module("ai4sci_bench.scorers." + module_path.stem)
    loader = importlib.import_module("ai4sci_bench.scorers.custom")
    task_dir = task_dir.resolve()
    # Reject cached local helpers from another task before executing task code.
    for path in task_dir.glob("*.py"):
        module = sys.modules.get(path.stem)
        if module is not None:
            location = getattr(module, "__file__", None)
            if not location or Path(location).resolve() != path:
                raise ValueError("Task helper import origin mismatch")
    loader.load_custom_scorer(task_dir)
    missing = requested - set(registry.list_scorers())
    if missing:
        raise ValueError(f"Scorer registry mismatch: unregistered scorers {sorted(missing)}")
    classes = {}
    for name in sorted(requested):
        scorer = registry.get_scorer(name)
        if not isinstance(scorer, registry.Scorer):
            raise ValueError(f"Scorer did not inherit native Scorer: {name}")
        classes[name] = type(scorer).__name__
    return {"scorers": classes, "modules": check_evaluator_imports(evaluator_root),
            "task_scorer": str(task_dir / "custom_scorer.py")
                if (task_dir / "custom_scorer.py").is_file() else None}


def _load_regular_json(path: Path, *, label: str, limit: int = 16 * 1024 * 1024) -> Any:
    """Read a bounded regular JSON file without following a final symlink."""
    path = Path(path)
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds size limit: {path}")
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


# Restricted container contract, not the host scoring CLI.
_INPUT_PREFIXES = ('task_bundle/', 'instance/', 'prediction/')


def _input_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Never follow links or open special files in the received snapshot."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError('unsafe input root')
    files = {}
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('unsafe input file')
        name = path.relative_to(root).as_posix()
        if name == 'evaluation_manifest.json':
            continue
        if not name.startswith(_INPUT_PREFIXES) and name != 'instance_parameters.json':
            raise ValueError('unexpected input path')
        data = path.read_bytes()
        files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return files


def load_evaluation_request(manifest_path: Path) -> dict[str, Any]:
    """Verify the received binding; host authenticity remains the caller's duty."""
    import re

    if manifest_path.name != 'evaluation_manifest.json' or manifest_path.is_symlink():
        raise ValueError('unexpected request path')
    request = _load_regular_json(manifest_path, label='evaluation request')
    if not isinstance(request, dict) or set(request) != {
            'schema_version', 'official', 'seed', 'task_id', 'instance_id', 'prompt_level',
            'attempt', 'provenance', 'runtime', 'files'}:
        raise ValueError('unexpected request fields')
    if (request.get('schema_version') != 2 or request.get('official') is not False
            or type(request.get('seed')) is not int
            or any(not isinstance(request.get(key), str) or not request[key].strip()
                   for key in ('task_id', 'instance_id', 'prompt_level'))):
        raise ValueError('invalid evaluation identity')
    attempt = request['attempt']
    if not isinstance(attempt, dict) or set(attempt) != {
            'number', 'attempt_status', 'collection_status', 'prediction_status', 'persisted_outputs'}:
        raise ValueError('unexpected attempt fields')
    for key, expected in [('attempt_status', 'completed'), ('collection_status', 'collected'),
                          ('prediction_status', 'present')]:
        if attempt.get(key) != expected:
            raise ValueError('ineligible attempt')
    if type(attempt.get('number')) is not int or attempt['number'] < 1:
        raise ValueError('invalid attempt number')
    provenance = request['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('invalid provenance')
    for key in ('result_sha256', 'native_result_sha256', 'prepared_manifest_sha256',
                'source_manifest_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', provenance.get(key, '')):
            raise ValueError('missing provenance digest: ' + key)
    sources = provenance.get('sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('missing source identities')
    for name, source in sources.items():
        if (not isinstance(name, str) or not name or not isinstance(source, dict)
                or set(source) != {'repo', 'path', 'resolved_revision'}
                or any(not isinstance(source[key], str) or not source[key].strip()
                       for key in ('repo', 'path', 'resolved_revision'))
                or not re.fullmatch('[0-9a-f]{40}', source['resolved_revision'])):
            raise ValueError('invalid source identity')
    for key in ('agent', 'model', 'requested_effort'):
        if not isinstance(provenance.get(key), (str, type(None))):
            raise ValueError('invalid agent provenance')
    # Do not echo arbitrary caller keys, paths or credentials into the report.
    allowed = {'result_sha256', 'native_result_sha256', 'prepared_manifest_sha256',
               'source_manifest_sha256', 'sources', 'agent', 'model', 'requested_effort'}
    if set(provenance) != allowed:
        raise ValueError('unexpected provenance fields')
    runtime = request['runtime']
    if set(runtime) != {'evaluator_bundle_digest', 'bridge_sha256', 'launcher_sha256',
                        'image_id'}:
        raise ValueError('unexpected runtime identity')
    for key, value in runtime.items():
        pattern = 'sha256:[0-9a-f]{64}' if key == 'image_id' else '[0-9a-f]{64}'
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError('invalid runtime digest')
    root = manifest_path.parent
    actual = _input_inventory(root)
    if actual != request['files']:
        raise ValueError('input inventory/digest mismatch')
    required = {'instance/instance_meta.json', 'instance_parameters.json'}
    if not required <= actual.keys():
        raise ValueError('missing required scoring input')
    # Task-specific files are checked by the task's gates, not by this bridge.
    resolve_task_sources(root / 'task_bundle')
    for directory in ('prediction', 'instance/reference'):
        if not (root / directory).is_dir():
            raise ValueError('missing scoring directory: ' + directory)
    persisted = attempt['persisted_outputs']
    if persisted != {p.removeprefix('prediction/'): v for p, v in actual.items() if p.startswith('prediction/')}:
        raise ValueError('persisted outputs binding mismatch')
    params = json.loads((root / 'instance_parameters.json').read_text())
    instance = json.loads((root / 'instance/instance_meta.json').read_text())
    if (not isinstance(instance, dict) or not isinstance(params, dict)
            or instance.get('task_id') != request['task_id']
            or params != (instance.get('params_used') or {})
            or any(key in instance and instance[key] != request[key]
                   for key in ('instance_id', 'seed', 'prompt_level'))):
        raise ValueError('instance parameter mismatch')
    return request


def build_evaluation_report(request, gates, scores, score, max_score):
    """Preserve native details; ambiguous setup failures are not infrastructure claims."""
    import dataclasses
    import math

    if (isinstance(score, bool) or isinstance(max_score, bool)
            or not math.isfinite(score) or not math.isfinite(max_score)
            or max_score <= 0 or not 0 <= score <= max_score):
        raise ValueError('invalid aggregate score/max_score')
    gate_results = [dataclasses.asdict(d) for d in gates]
    score_details = [dataclasses.asdict(d) for d in scores]
    error = None
    for item in gate_results + score_details:
        if not math.isfinite(item['score']) or not math.isfinite(item['max_score']):
            raise ValueError('invalid ScoreDetail numeric value')
        details = item['details'] or {}
        if details.get('scorer_internal_error'):
            error = {'category': 'scorer_internal_error', 'scorer': item['scorer_name']}
            break
        if details.get('setup_error'):
            error = {'category': 'scorer_setup_unclassified', 'scorer': item['scorer_name']}
    return {
        'schema_version': 1, 'official': False,
        **{k: request[k] for k in ('task_id', 'instance_id', 'seed', 'prompt_level')},
        'attempt_status': request['attempt']['attempt_status'],
        'evaluation_status': 'failed' if error else 'completed',
        'score': None if error else score, 'max_score': max_score,
        'reward': None if error else score / max_score,
        'gate_results': gate_results, 'score_details': score_details,
        'error': error, 'provenance': request['provenance'],
        'attempt': request['attempt'], 'input_files': request.get('files', {}),
    }


def verify_scoring_module(evaluator_manifest: dict) -> None:
    """Bind the imported local implementation to the verified evaluator manifest."""
    import scoring

    path = Path(__file__).with_name("scoring.py")
    if (path.is_symlink() or Path(scoring.__file__).resolve() != path.resolve()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != evaluator_manifest.get("scoring", {}).get("sha256")):
        raise ValueError("scoring module identity mismatch")


def evaluate_request(request: dict, input_root: Path) -> dict:
    """Container only: bind image and inputs, then run the unchanged aggregation."""
    import time

    started = time.monotonic()
    expected = request['runtime']
    for path, key in [(Path(__file__), 'bridge_sha256'),
                      (Path('/opt/bridge/test.sh'), 'launcher_sha256')]:
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise ValueError('runtime file identity mismatch: ' + key)
    evaluator_manifest = _load_regular_json(Path('/opt/asi-evaluator/evaluator_manifest.json'), label='evaluator manifest')
    if evaluator_manifest.get('launcher', {}).get('sha256') != expected['launcher_sha256']:
        raise ValueError('launcher evaluator binding mismatch')
    verify_scoring_module(evaluator_manifest)
    task_dir = input_root / 'task_bundle'
    metadata = load_scoring_metadata(task_dir)
    if (metadata.get('id') != request['task_id']
            or evaluator_manifest.get('bundle_digest') != expected['evaluator_bundle_digest']):
        raise ValueError('evaluator request binding mismatch')
    registration = register_task_scorers(task_dir, request['task_id'], metadata['evaluation'], Path('/opt/asi-evaluator'))
    evaluation = metadata['evaluation']
    parameters = json.loads((input_root / 'instance_parameters.json').read_text())
    gates, hard_pass, soft_failures, scores, score = _evaluate_gates_and_scores(
        evaluation, Path('/prediction'), input_root / 'instance/reference', parameters, prompt_level=request['prompt_level'])
    maximum = sum(item.get('weight', 1.0) for item in evaluation['scoring'])
    report = build_evaluation_report(request, gates, scores, score, maximum)
    report.update({'hard_gates_passed': hard_pass, 'soft_gate_failures': soft_failures,
                   'elapsed_seconds': time.monotonic() - started})
    if _input_inventory(input_root) != request['files']:
        raise ValueError('input changed during scoring')
    report['provenance'] = {**report['provenance'], 'runtime': expected,
                            'registration': registration, 'evaluator': evaluator_manifest,
                            'input_tree_sha256': hashlib.sha256(json.dumps(request['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                            'request_sha256': hashlib.sha256((input_root / 'evaluation_manifest.json').read_bytes()).hexdigest()}
    return report


def _json_safe(value):
    """Native diagnostics may contain inf; retain an explicit string, never JSON NaN."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_evaluation_result(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError('unsafe output directory')
    score_path = output_dir / 'asibench_score.json'
    reward_path = output_dir / 'reward.txt'
    if score_path.exists() or score_path.is_symlink() or reward_path.exists() or reward_path.is_symlink():
        raise FileExistsError('refusing existing evaluation output')
    # Exclusive creation: never truncate a previous result, even on failure.
    with score_path.open('x') as stream:
        stream.write(json.dumps(_json_safe(report), indent=2, allow_nan=False) + '\n')
    if report['evaluation_status'] == 'completed':
        with reward_path.open('x') as stream:
            stream.write(str(report['reward']) + '\n')


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Restricted ASI scoring container bridge')
    parser.add_argument('--manifest', type=Path, default=Path('/input/evaluation_manifest.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('/output'))
    args = parser.parse_args()
    if args.manifest != Path('/input/evaluation_manifest.json') or args.output_dir != Path('/output'):
        parser.error('container paths are fixed to /input/evaluation_manifest.json and /output')
    phase = 'input_preflight'
    request = None
    try:
        if os.getuid() != 10001 or os.getgid() != 10001:
            raise ValueError('bridge requires UID/GID 10001')
        request = load_evaluation_request(args.manifest)
        phase = 'runtime_or_scoring'
        report = evaluate_request(request, args.manifest.parent)
    except Exception as exc:
        logger.exception('Evaluation failed in %s', phase)
        report = {'schema_version': 1, 'official': False, 'evaluation_status': 'failed',
                  'score': None, 'max_score': None, 'reward': None, 'gate_results': [], 'score_details': [],
                  'error': {'category': phase, 'exception_type': type(exc).__name__, 'message': str(exc)}}
        if request is not None:
            report.update({k: request[k] for k in ('task_id', 'instance_id', 'seed', 'prompt_level')})
            report['attempt_status'] = request['attempt']['attempt_status']
            report['provenance'] = request['provenance']
    write_evaluation_result(report, args.output_dir)
    return 0 if report['evaluation_status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
