"""Read-only recovery planning and native orchestrator dispatch (no experiment paths)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from ai4sci_bench.runner.claude_checkpoint import checkpoint_key
from ai4sci_bench.runner.session_checkpoint import atomic_json, fingerprint, inventory, load_checkpoint


def _selection(value):
    if value in (None, 'all'):
        return None
    values = value.split(',')
    if not all(values) or len(set(values)) != len(values):
        raise ValueError('Selections must be comma-separated unique nonempty values')
    return set(values)


def plan_resume(source, tasks=None, levels=None, instance_ids=None):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif (source / 'checkpoint.json').is_file():
        paths = [source / 'checkpoint.json']
    else:
        paths = sorted((source / 'checkpoints').glob('*/checkpoint.json'))
    if not paths:
        raise ValueError('No native Claude checkpoints; old trajectory logs alone cannot restore a session')
    task_set, level_set, instance_set = map(_selection, (tasks, levels, instance_ids))
    if level_set and level_set - {'b1', 'b2', 'b3', 'b4'}:
        raise ValueError('Levels must be b1,b2,b3,b4')
    chosen, rejected, seen = {}, [], set()
    for path in paths:
        data = json.loads(path.read_text())
        config = data.get('config') or {}
        task, level, instance = config.get('task_id'), config.get('level'), config.get('instance_id')
        if any(selected is not None and value not in selected for selected, value in (
                (task_set, task), (level_set, level), (instance_set, instance))):
            continue
        key = checkpoint_key(task, instance, level)
        if key in seen:
            raise ValueError('Duplicate checkpoint for the same task/instance/level')
        seen.add(key)
        try:
            verified, data = load_checkpoint(path)
            if path.with_name('continuation.json').exists():
                raise ValueError('Checkpoint already has a continuation; use the child checkpoint')
            chosen[key] = (verified, data)
        except (ValueError, OSError, KeyError) as exc:
            rejected.append({'cell': key, 'reason': str(exc)})
    selected_configs = [json.loads(p.read_text())['config'] for p in paths]
    # Missing explicit Cartesian task/level pairs must not silently disappear.
    if task_set and level_set:
        present = {(c.get('task_id'), c.get('level')) for c in selected_configs
                   if instance_set is None or c.get('instance_id') in instance_set}
        if {(t, l) for t in task_set for l in level_set} - present:
            raise ValueError('Some explicitly selected task/level pairs have no checkpoint')
    for selected, field in ((task_set, 'task_id'), (level_set, 'level'), (instance_set, 'instance_id')):
        if selected and selected - {c.get(field) for c in selected_configs}:
            raise ValueError(f'Selected {field} has no checkpoint')
    return chosen, rejected


def preflight_inputs(chosen, tasks_dir, instances_dir=None):
    """Prepare only public agent inputs in disposable directories; never generate GT."""
    from ai4sci_bench.core.task import TaskLoader
    from ai4sci_bench.core.types import PromptLevel
    from ai4sci_bench.generators.instance_generator import InstanceGenerator
    loader = TaskLoader(Path(tasks_dir))
    if instances_dir is None:
        parents = {str(Path(data['instance_bundle']).parent) for _, data in chosen.values()}
        if len(parents) != 1:
            raise ValueError('Provide --instances-dir for the original materialized bundles')
        instances_dir = next(iter(parents))
    instances_root = Path(instances_dir).expanduser().resolve()
    generator = InstanceGenerator(Path(tasks_dir), sandbox='os')
    for _, data in chosen.values():
        config = data['config']
        from ai4sci_bench.runner.claude_checkpoint import implementation_identity
        implementation = config.get('implementation')
        if not isinstance(implementation, dict) or implementation != implementation_identity():
            raise ValueError('Checkpoint framework implementation differs from this installation')
        task = loader.load_task_by_id(config['task_id'])
        if fingerprint({k: v for k, v in task.items() if not k.startswith('_')}) != config['task_definition']:
            raise ValueError('Task definition changed; cannot continue the same experiment')
        bundle = instances_root / config['instance_id']
        if not bundle.is_dir():
            raise ValueError('Original instance bundle missing; provide --instances-dir')
        metadata = bundle / 'instance_meta.json'
        actual_parameters = json.loads(metadata.read_text()).get('params_used', {}) if metadata.is_file() else {}
        if actual_parameters != config['parameters']:
            raise ValueError('Original instance parameters changed; cannot resume')
        with tempfile.TemporaryDirectory(prefix='asibench-resume-preflight-') as temporary:
            workspace = Path(temporary) / 'workspace'
            workspace.mkdir()
            generator._prepare_workspace(bundle, workspace, config['instance_id'], PromptLevel(config['level']),
                task, config['parameters'], effective_timeout_seconds=data['budget']['limit_seconds'],
                framework_task_info_dir=Path(temporary) / 'framework')
            if inventory(workspace) != config['input_files']:
                raise ValueError('Original prompt/input files changed; cannot resume')
    return instances_root


def execute_resume(chosen, output, tasks_dir, instances_dir, parallel=1, api_key_env=None, api_base_env=None):
    from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
    from ai4sci_bench.runner.parallel import auto_limit_workers
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(path.parent) for path, _ in chosen.values()):
        raise ValueError('Resume output must be a new directory outside original checkpoints')
    profile_keys = ('model', 'effort', 'timeout_seconds', 'tool_mode', 'permission_mode',
                    'supports_image_input', 'token_prices_usd_per_million', 'implementation')
    profiles = {fingerprint({k: data['config'][k] for k in profile_keys}) for _, data in chosen.values()}
    if len(profiles) != 1:
        raise ValueError('Selected checkpoints have different experiment profiles; resume separately')
    data = next(iter(chosen.values()))[1]
    config, connection = data['config'], data['connection']
    base_selector = api_base_env or connection.get('api_base_env')
    adapter = ClaudeCodeCLIAdapter(model=config['model'], effort=config['effort'],
        permission_mode=config['permission_mode'], api_protocol='anthropic',
        supports_image_input=config['supports_image_input'],
        api_key_env=api_key_env or connection['api_key_env'],
        api_base_env=base_selector, api_base=None if base_selector else connection['api_base'],
        network_policy='model-only', tool_mode=config['tool_mode'])
    run_config = RunConfig(agent=adapter, tasks=sorted({d['config']['task_id'] for _, d in chosen.values()}),
        prompt_levels=sorted({d['config']['level'] for _, d in chosen.values()}),
        instances_dir=str(instances_dir), tasks_dir=str(tasks_dir), output_dir=str(output),
        include_test=True, include_sample=True, include_dev=True, include_abandoned=True,
        sandbox='os', timeout=config['timeout_seconds'], parallel=auto_limit_workers(parallel, sandbox='os'),
        checkpoint_sessions=True, resume_checkpoints={key: str(path) for key, (path, _) in chosen.items()},
        checkpoint_token_prices=config['token_prices_usd_per_million'], retries=1, score=False,
        agent_metadata={'adapter': 'claude_code_cli', 'model': config['model'], 'effort': config['effort'],
                        'recovery_mode': 'native-session'})
    orchestrator = BenchmarkOrchestrator(run_config)
    atomic_json(output / 'resume-plan.json', {key: str(path) for key, (path, _) in chosen.items()})
    return orchestrator.run()
