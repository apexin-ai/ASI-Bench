"""Bridge contracts and adapter-owned scoring behavior tests.

Set ASI_BENCH_SOURCE to a local Git checkout containing the pinned revision.
Missing source fails explicitly. Only hash-checked trusted framework code and
synthetic scorers execute; no task scorer, prediction or full package import.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REVISION = '5935b5f33549e348a8505bb355d3b6f4fe4a273c'
BRIDGE = Path(__file__).resolve().parents[1] / 'templates/verifier/score_entry.py'
SCORING = BRIDGE.with_name('scoring.py')
SOURCES = {
    'scorers/custom.py': 'a62a740716f549449bf2305646aff952587f2e54a52bf5511c8395a8eb8eb742',
    'core/task.py': '9418656108a7c9007f52f1255bbdbab68a4ad889c260086aa62eec340d62e62b',
    'core/types.py': '91130a9bd04b7ae656e24b40b0038524dd6034b4470229b988a5531e32e80cc3',
    'core/scorer.py': 'd7f3aed05e7ce26629827f2a15ddc81ed1cd7fd11cbe7dcfaf8ab7cb35505417',
    'core/logger.py': 'ed86ad13814d7401358b42a994b0744d44fa60bbf97630aa374b8b418d06d1a4',
}
FUNCTIONS = {
    'core/task.py': ('resolve_task_sources',),
}
CONSTANTS = {'LEGACY_TASK_FILE': 'task.yaml', 'META_TASK_FILE': 'task_meta.yaml',
             'EVAL_TASK_FILE': 'task_eval.yaml'}


@pytest.fixture(scope='module')
def source_files():
    root = os.environ.get('ASI_BENCH_SOURCE')
    assert root, 'Set ASI_BENCH_SOURCE to the ASI checkout; P1.1 source verification is required'
    files = {}
    for path, digest in SOURCES.items():
        data = subprocess.run(
            ['git', '-C', root, 'show', f'{REVISION}:ai4sci_bench/{path}'],
            check=True, capture_output=True, timeout=15,
        ).stdout
        assert hashlib.sha256(data).hexdigest() == digest, path
        files[path] = data.decode()
    return files


def test_task_source_extraction_and_restricted_imports(source_files):
    bridge = ast.parse(BRIDGE.read_text())
    actual = {n.name: n for n in bridge.body if isinstance(n, ast.FunctionDef)}
    for path, names in FUNCTIONS.items():
        original = {n.name: n for n in ast.parse(source_files[path]).body
                    if isinstance(n, ast.FunctionDef)}
        for name in names:
            assert ast.dump(actual[name]) == ast.dump(original[name]), name
    for name, value in CONSTANTS.items():
        for tree in (bridge, ast.parse(source_files['core/task.py'])):
            assignments = {n.targets[0].id: ast.literal_eval(n.value)
                           for n in tree.body if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name) and n.targets[0].id in CONSTANTS}
            assert assignments[name] == value
    allowed = {'__future__', 'ast', 'hashlib', 'json', 'os', 'stat', 'subprocess', 'traceback',
               'pathlib', 'typing', 'yaml', 'sys', 'importlib', 'argparse', 'dataclasses', 'math', 're', 'time',
               'ai4sci_bench.core.logger', 'ai4sci_bench.core.scorer', 'ai4sci_bench.core.types',
               'ai4sci_bench.runner.task_env', 'scoring'}
    for node in [*ast.walk(bridge), *ast.walk(ast.parse(SCORING.read_text()))]:
        if isinstance(node, ast.Import):
            assert all(item.name in allowed for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed




@pytest.mark.parametrize('group', ['config', 'templates', 'aggregation',
                                  'register-ok', 'register-wrapped', 'register-fallback', 'register-foreign',
                                  'register-unsupported', 'register-wrong-base', 'generic-scorer', 'bridge-result', 'bridge-input', 'scoring-identity'])
def test_isolated_contract(source_files, tmp_path, group):
    # Exercise deployed siblings, without importing from the repository templates.
    deployed = tmp_path / 'bridge'
    deployed.mkdir()
    for template in (BRIDGE, SCORING):
        (deployed / template.name).write_bytes(template.read_bytes())
    # Temporary test fixture only, not the P1.2 production bundle materializer.
    package = tmp_path / 'ai4sci_bench/core'
    package.mkdir(parents=True)
    (package.parent / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    for name in ('types.py', 'scorer.py', 'logger.py'):
        (package / name).write_text(source_files[f'core/{name}'])
    (tmp_path / 'sources.json').write_text(json.dumps(source_files))
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-I', str(Path(__file__).resolve()), group, str(tmp_path), str(deployed / BRIDGE.name)],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_scoring_identity(bridge, local_scoring, bridge_path):
    """Guards local-module extraction from the adapter baseline at commit 9b90d16c."""
    path = bridge_path.with_name('scoring.py')
    manifest = {'scoring': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    bridge.verify_scoring_module(manifest)
    for invalid in ({}, {'scoring': {'sha256': '0' * 64}}):
        with pytest.raises(ValueError, match='scoring module identity mismatch'):
            bridge.verify_scoring_module(invalid)
    original = local_scoring.__file__
    try:
        local_scoring.__file__ = str(path.with_name('foreign.py'))
        with pytest.raises(ValueError, match='scoring module identity mismatch'):
            bridge.verify_scoring_module(manifest)
    finally:
        local_scoring.__file__ = original
    path.write_bytes(path.read_bytes() + b'\n# changed after deployment\n')
    with pytest.raises(ValueError, match='scoring module identity mismatch'):
        bridge.verify_scoring_module(manifest)
    target = path.with_name('relocated.py')
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match='scoring module identity mismatch'):
        bridge.verify_scoring_module(manifest)


def run_contract(group, root, bridge_path):
    import copy
    import importlib.util
    import types
    from dataclasses import asdict

    import yaml

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(bridge_path.parent))
    import scoring as local_scoring
    from ai4sci_bench.core import scorer as registry
    from ai4sci_bench.core.logger import get_logger
    from ai4sci_bench.core.types import GenerationMode, PromptLevel, ScoreDetail
    assert Path(registry.__file__).resolve().is_relative_to(root)
    spec = importlib.util.spec_from_file_location('score_entry', bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    if group == 'scoring-identity':
        check_scoring_identity(bridge, local_scoring, bridge_path)
        return
    sources = json.loads((root / 'sources.json').read_text())
    # Execute only pinned AST-selected definitions, never orchestrator imports.
    baseline = types.ModuleType('baseline')
    from typing import Any
    baseline.__dict__.update(ast=ast, Path=Path, Any=Any,
                             yaml=yaml, PromptLevel=PromptLevel, ScoreDetail=ScoreDetail,
                             GenerationMode=GenerationMode, get_scorer=registry.get_scorer,
                             logger=get_logger('p11.baseline'), **CONSTANTS)
    for path, names in FUNCTIONS.items():
        if path == 'core/task.py':
            names += ('_merge_output_files', 'merge_task_eval')
        tree = ast.parse(sources[path])
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), baseline.__dict__)
    task_tree = ast.parse(sources['core/task.py'])
    loader = next(n for n in task_tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskLoader')
    method = next(n for n in loader.body if isinstance(n, ast.FunctionDef) and n.name == 'load_task_metadata')
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'core/task.py', 'exec'), baseline.__dict__)

    if group == 'generic-scorer':
        # Guards generic-entry work against the MPSC-only baseline at commit 9b90d16c.
        scorers = root / 'ai4sci_bench/scorers'
        scorers.mkdir()
        (scorers / '__init__.py').write_text('')
        (scorers / 'file_match.py').write_text('')
        (scorers / 'custom.py').write_text(sources['scorers/custom.py'])
        task = root / 'task'
        task.mkdir()
        (task / 'custom_scorer.py').write_text(
            'from ai4sci_bench.core.scorer import Scorer, register_scorer\n'
            'from ai4sci_bench.core.types import ScoreDetail\n'
            '@register_scorer("text_answer")\n'
            'class TextAnswer(Scorer):\n'
            '    def score(self, pred_dir, ref_dir, config):\n'
            '        assert config["prompt_level"] == "b2"\n'
            '        passed = (pred_dir / "answer.txt").read_text() == "42"\n'
            '        return ScoreDetail("text_answer", config["weight"] if passed else 0, '
            'config["weight"], passed, {}, "checked")\n')
        evaluation = {'scoring': [{'scorer': 'text_answer', 'weight': 5}]}
        registration = bridge.register_task_scorers(task, 'text.example', evaluation, root)
        assert registration['scorers'] == {'text_answer': 'TextAnswer'}
        assert 'mpsc_eval_runtime' not in sys.modules
        prediction = root / 'prediction'
        prediction.mkdir()
        (prediction / 'answer.txt').write_text('42')
        gates, hard, soft, scores, total = bridge._evaluate_gates_and_scores(
            evaluation, prediction, root / 'reference', {}, prompt_level='b2')
        assert hard and soft == 0 and not gates and total == 5
        assert scores[0].passed
        (prediction / 'answer.txt').write_text('wrong')
        assert bridge._evaluate_gates_and_scores(
            evaluation, prediction, root / 'reference', {}, prompt_level='b2')[-1] == 0
        return

    if group == 'bridge-input':
        inputs = root / 'input'
        inputs.mkdir()
        names = ['task_bundle/' + name for name in ('task_meta.yaml', 'task_eval.yaml', 'custom_scorer.py', 'mpsc_eval_runtime.py')]
        names += ['instance/' + name for name in ('instance_meta.json', 'data/system.json', 'data/public_cases.json',
            'reference/certificate.json', 'reference/hidden_cases.json', 'reference/data/system.json', 'reference/data/public_cases.json')]
        names += ['instance_parameters.json', 'prediction/analysis.py', 'prediction/certificate.json']
        for name in names:
            path = inputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        (inputs / 'instance/instance_meta.json').write_text(json.dumps({'task_id': 'math.mpsc_safety_filter', 'seed': 31415,
            'instance_id': 'math.mpsc_safety_filter__seed31415', 'prompt_level': 'b1'}))
        files = bridge._input_inventory(inputs)
        request = {'schema_version': 2, 'official': False, 'seed': 31415, 'task_id': 'math.mpsc_safety_filter',
            'instance_id': 'math.mpsc_safety_filter__seed31415', 'prompt_level': 'b1',
            'attempt': {'number': 1, 'attempt_status': 'completed', 'collection_status': 'collected',
                'prediction_status': 'present', 'persisted_outputs': {p.removeprefix('prediction/'): v for p, v in files.items() if p.startswith('prediction/')}},
            'provenance': {**dict.fromkeys(('result_sha256', 'native_result_sha256', 'prepared_manifest_sha256', 'source_manifest_sha256'), 'a'*64),
                'agent': 'test', 'model': 'test', 'requested_effort': None, 'sources': {
                    'hf': {'repo': 'Apexintelligence-AI/ASI-Bench-seed31415', 'path': 'tasks/math.mpsc_safety_filter__seed31415', 'resolved_revision': 'a'*40},
                    'github': {'repo': 'apexin-ai/ASI-Bench', 'path': 'tasks/math/mpsc_safety_filter', 'resolved_revision': 'a797bf69683400bed0f38c4add507bb3d404b833'}}},
            'runtime': {**dict.fromkeys(('evaluator_bundle_digest', 'bridge_sha256', 'launcher_sha256'), 'a'*64), 'image_id': 'sha256:'+'a'*64},
            'files': files}
        manifest = inputs / 'evaluation_manifest.json'
        manifest.write_text(json.dumps(request))
        assert bridge.load_evaluation_request(manifest) == request
        for fields, value in [
            (('schema_version',), 1),
            (('seed',), 42), (('instance_id',), 'math.mpsc_safety_filter__seed42'),
            (('prompt_level',), ''), (('task_id',), 'other'),
            (('attempt', 'attempt_status'), 'execution_failed'),
            (('attempt', 'collection_status'), 'failed'), (('attempt', 'prediction_status'), 'missing'),
            (('attempt', 'persisted_outputs'), {}), (('runtime', 'bridge_sha256'), 'bad'),
            (('provenance', 'result_sha256'), 'bad'), (('files',), {'../escape': {}}),
            (('provenance', 'api_key'), 'must-not-echo')]:
            bad = copy.deepcopy(request)
            target = bad
            for field in fields[:-1]:
                target = target[field]
            target[fields[-1]] = value
            manifest.write_text(json.dumps(bad))
            try:
                bridge.load_evaluation_request(manifest)
            except (ValueError, KeyError):
                pass
            else:
                raise AssertionError(('bad request accepted', fields))
        manifest.write_text(json.dumps(request))
        hidden = inputs / 'instance/reference/hidden_cases.json'
        hidden.unlink()
        try:
            bridge.load_evaluation_request(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('missing reference accepted')
        hidden.symlink_to(inputs / 'prediction/certificate.json')
        try:
            bridge.load_evaluation_request(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('symlink accepted')
        hidden.unlink()
        os.mkfifo(hidden)
        try:
            bridge.load_evaluation_request(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('FIFO accepted')

    if group == 'bridge-input':
        # Guards generic-entry work against the MPSC-only baseline at commit 9b90d16c.
        import shutil
        shutil.rmtree(inputs)
        (inputs / 'task_bundle').mkdir(parents=True)
        (inputs / 'instance/reference').mkdir(parents=True)
        (inputs / 'prediction').mkdir()
        (inputs / 'task_bundle/task.yaml').write_text('id: text.example\n')
        (inputs / 'prediction/answer.txt').write_text('42')
        (inputs / 'instance_parameters.json').write_text('{"n": 2}')
        identity = {'task_id': 'text.example', 'instance_id': 'sample-A', 'seed': 42,
                    'prompt_level': 'b2'}
        (inputs / 'instance/instance_meta.json').write_text(json.dumps({
            **identity, 'params_used': {'n': 2}}))
        request.update(identity)
        request['provenance']['sources'] = {'github': {
            'repo': 'example/tasks', 'path': 'text/example', 'resolved_revision': 'b' * 40}}
        request['files'] = bridge._input_inventory(inputs)
        request['attempt']['persisted_outputs'] = {'answer.txt': request['files']['prediction/answer.txt']}
        manifest.write_text(json.dumps(request))
        assert bridge.load_evaluation_request(manifest) == request
        (inputs / 'prediction/answer.txt').write_text('tampered')
        with pytest.raises(ValueError, match='inventory/digest mismatch'):
            bridge.load_evaluation_request(manifest)

    if group == 'bridge-result':
        request = {'task_id': 'math.mpsc_safety_filter', 'instance_id': 'math.mpsc_safety_filter__seed31415',
                   'seed': 31415, 'prompt_level': 'b1', 'attempt': {'attempt_status': 'completed'},
                   'provenance': {}}
        gate = ScoreDetail('file_match', 0.0, 1.0, False, {}, 'missing')
        report = bridge.build_evaluation_report(request, [gate], [], 0.0, 100.0)
        assert report['evaluation_status'] == 'completed' and report['reward'] == 0.0
        bridge.write_evaluation_result(report, root / 'out')
        assert (root / 'out/reward.txt').read_text().strip() == '0.0'
        gate.details = {'scorer_internal_error': True}
        failed = bridge.build_evaluation_report(request, [gate], [], 0.0, 100.0)
        assert failed['evaluation_status'] == 'failed' and failed['reward'] is None
        assert failed['error']['category'] == 'scorer_internal_error'
        gate.details = {'setup_error': 'unknown origin'}
        assert bridge.build_evaluation_report(request, [gate], [], 0.0, 100.0)['error']['category'] == 'scorer_setup_unclassified'
        for score, maximum in [(float('nan'), 100), (1, 0), (101, 100), (-1, 100)]:
            try:
                bridge.build_evaluation_report(request, [], [], score, maximum)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid numeric result accepted')
        try:
            bridge.write_evaluation_result(failed, root / 'out')
        except FileExistsError:
            pass
        else:
            raise AssertionError('old reward overwritten/reused')
        bridge.write_evaluation_result(failed, root / 'failed')
        assert not (root / 'failed/reward.txt').exists()

    def outcome(fn, *args, **kwargs):
        try:
            return ('ok', fn(*args, **kwargs))
        except Exception as exc:
            return ('error', type(exc).__name__, str(exc))

    if group == 'config':
        cases = [
            {'task.yaml': 'id: old\nevaluation: {gates: []}\n'},
            {'task_meta.yaml': 'id: meta\n'},
            {'task_meta.yaml': 'id: meta\noutput:\n  files: [{name: x, type: json}]\n',
             'task_eval.yaml': 'task_id: other\nevaluation: {scoring: []}\noutput:\n  files: [{name: x, shape: [2]}, {name: y}]\n'},
            {'task.yaml': 'id: legacy\n', 'task_meta.yaml': 'id: split\n', 'task_eval.yaml': 'evaluation: {}'},
            {}, {'task_meta.yaml': '['},
            {'task_meta.yaml': 'id: meta', 'task_eval.yaml': '['},
            {'task_meta.yaml': 'id: meta', 'task_eval.yaml': '[]'},
        ]
        for i, files in enumerate(cases):
            task = root / f'config-{i}'
            task.mkdir()
            for name, text in files.items():
                (task / name).write_text(text)
            actual = outcome(bridge.load_scoring_metadata, task)
            expected = outcome(baseline.load_task_metadata, None, task / 'task.yaml')
            if expected[0] == 'ok':
                expected = ('ok', {k: v for k, v in expected[1].items() if k in {'id', 'runtime', 'evaluation'}})
            assert actual == expected, (i, actual, expected)
            if i == 2:
                assert actual[1]['id'] == 'meta'
                assert 'output' not in actual[1]
            if i == 3:
                assert actual[1]['id'] == 'legacy'
        # Explicit adaptation: no generation/runtime bookkeeping or validation.
        task = root / 'projection'
        task.mkdir()
        (task / 'task_meta.yaml').write_text('id: demo\ngeneration: {mode: invalid}\nruntime: {packages: invalid}\n')
        assert not any(k.startswith('_') for k in bridge.load_scoring_metadata(task))
        assert outcome(baseline.load_task_metadata, None, task / 'task.yaml')[0] == 'error'
    elif group == 'templates':
        cases = [
            ({'a': ['{n+1}', '{n, n*2}', 'text', 3]}, {'a': [6, [5, 10], 'text', 3]}),
            ('{-n}', -5), ('{n/2}', 2.5), ('{n//2}', 2), ('{n%2}', 1), ('{+n}', 5),
        ]
        for value, expected in cases:
            unchanged = copy.deepcopy(value)
            assert local_scoring._resolve_config_templates(value, {'n': 5}) == expected
            assert value == unchanged
        for value, error in [('{missing}', KeyError), ('{n**2}', ValueError), ('{n/0}', ZeroDivisionError)]:
            with pytest.raises(error):
                local_scoring._resolve_config_templates(value, {'n': 5})
    elif group.startswith('register-'):
        scorers = root / 'ai4sci_bench/scorers'
        scorers.mkdir()
        (scorers / '__init__.py').write_text('')
        (scorers / 'file_match.py').write_text(
            'from ai4sci_bench.core.scorer import Scorer, register_scorer\n'
            '@register_scorer("file_match")\n'
            'class FileMatch(Scorer):\n'
            '    def score(self, *args): raise AssertionError("must not score")\n')
        (scorers / 'custom.py').write_text(
            sources['scorers/custom.py'])
        task = root / 'task'
        task.mkdir()
        (task / 'mpsc_eval_runtime.py').write_text('')
        code = ('import sys\nfrom pathlib import Path\n'
                'sys.path.insert(0, str(Path(__file__).parent))\nimport mpsc_eval_runtime\n'
                'from ai4sci_bench.core.scorer import Scorer, register_scorer\n')
        if group == 'register-fallback':
            code += 'def register_scorer(name): return lambda cls: cls\n'
        if group == 'register-wrong-base':
            code += 'Scorer = object\n'
        for name in ('mpsc_interface_smoke', 'mpsc_safety_filter'):
            code += (f'@register_scorer("{name}")\nclass {name}(Scorer):\n'
                     '    def score(self, *args): raise AssertionError("must not score")\n')
        if group == 'register-wrapped':
            # Scorer inheritance/registry are authoritative, not wrapper co_filename.
            code += ('namespace = {}\n'
                     'exec(compile("def wrapped(self, *args): raise AssertionError", '
                     '"<trusted-wrapper>", "exec"), namespace)\n'
                     'mpsc_safety_filter.score = namespace["wrapped"]\n')
        (task / 'custom_scorer.py').write_text(code)
        evaluation = {'gates': [{'scorer': 'file_match'}, {'scorer': 'mpsc_interface_smoke'}],
                      'scoring': [{'scorer': 'mpsc_safety_filter'}]}
        if group == 'register-unsupported':
            for name in ('llm_judge', 'unknown'):
                with pytest.raises(ValueError, match='unregistered scorers'):
                    bridge.register_task_scorers(task, 'math.mpsc_safety_filter',
                                                 {'scoring': [{'scorer': name}]}, root)
            # Task IDs are no longer a scorer-registration allowlist.
            report = bridge.register_task_scorers(task, 'other.task', evaluation, root)
            assert set(report['scorers']) == {'file_match', 'mpsc_interface_smoke', 'mpsc_safety_filter'}
        elif group == 'register-foreign':
            with pytest.raises(ValueError, match='origin mismatch'):
                bridge.register_task_scorers(task, 'math.mpsc_safety_filter', evaluation, root / 'wrong')
            sys.modules['ai4sci_bench.generators'] = types.ModuleType('ai4sci_bench.generators')
            with pytest.raises(ValueError, match='Unsupported evaluator module'):
                bridge.check_evaluator_imports(root)
            del sys.modules['ai4sci_bench.generators']
        elif group in ('register-fallback', 'register-wrong-base'):
            with pytest.raises(ValueError, match=r'registry mismatch|inherit native'):
                bridge.register_task_scorers(task, 'math.mpsc_safety_filter', evaluation, root)
        else:
            report = bridge.register_task_scorers(task, 'math.mpsc_safety_filter', evaluation, root)
            assert set(report['scorers']) == {'file_match', 'mpsc_interface_smoke', 'mpsc_safety_filter'}
            assert report['task_scorer'] == str(task / 'custom_scorer.py')
    else:
        calls = []
        @registry.register_scorer('synthetic')
        class Synthetic(registry.Scorer):
            def score(self, pred_dir, ref_dir, config):
                calls.append(copy.deepcopy(config))
                if config.get('crash'):
                    raise RuntimeError('synthetic scorer failure')
                return ScoreDetail(self.name, config.get('points', config['weight']),
                                   config['weight'], config.get('pass', True), {})

        def gate(**config):
            return {'scorer': 'synthetic', 'config': config}

        scoring = [{'scorer': 'synthetic', 'weight': 10, 'config': {'points': 3}},
                   {'scorer': 'synthetic', 'config': {}}]
        cases = [
            ({'gates': [gate(weight=99, prompt_level='b4')], 'scoring': scoring}, 4, 3, True, 0),
            ({'gates': [gate(**{'pass': False}), gate()], 'scoring': scoring}, 0, 2, False, 0),
            ({'gates': [dict(gate(**{'pass': False}), severity='soft')], 'scoring': scoring}, 4, 3, True, 1),
            ({'gates': [gate(**{'pass': False, 'points': .25}), gate()], 'hard_fail_score_mode': 'gate_score_sum', 'scoring': scoring}, 1.25, 2, False, 0),
            ({'gates': [gate(crash=True)], 'scoring': scoring}, 0, 1, False, 0),
            ({'scoring': [gate(crash=True)]}, 0, 1, True, 0),
            ({}, 0, 0, True, 0),
        ]

        def normalize(value):
            if value[0] == 'error':
                return value
            gates, hard, soft, scores, total = value[1]
            details = [asdict(x) for x in gates + scores]
            for detail in details:
                if detail['details'].get('scorer_internal_error'):
                    tail = detail['details'].pop('traceback_tail')
                    assert 1 <= len(tail) <= 5
                    assert any('File ' in line for line in tail)
                    assert tail[-1] == 'RuntimeError: synthetic scorer failure'
                    assert detail['details']['exception_type'] == 'RuntimeError'
            return details, hard, soft, total

        for evaluation, total, count, hard, soft in cases:
            calls.clear()
            unchanged = copy.deepcopy(evaluation)
            result = outcome(local_scoring._evaluate_gates_and_scores, evaluation,
                             root / 'prediction', root / 'reference', {}, prompt_level=PromptLevel.B1)
            assert evaluation == unchanged
            assert result[0] == 'ok'
            assert result[1][1:3] == (hard, soft)
            assert result[1][4] == total
            assert len(calls) == count
            assert all(c['prompt_level'] == 'b1' for c in calls)
            assert all(c['weight'] == 1 for c in calls[:len(evaluation.get('gates', []))])
            normalize(result)  # Also validate bounded diagnostics on scorer exceptions.
            gates, _, _, scores, _ = result[1]
            assert len(gates) == len(evaluation.get('gates', []))
            assert len(scores) == (len(evaluation.get('scoring', [])) if hard else 0)
            for detail, config in zip(scores, evaluation.get('scoring', []), strict=False):
                assert detail.max_score == config.get('weight', 1)
        for evaluation, error in [
            ({'gates': [{'scorer': 'missing'}]}, 'KeyError'),
            ({'scoring': [{'scorer': 'missing'}]}, 'KeyError'),
            ({'gates': [gate(points='{unknown}')]}, 'KeyError'),
            ({'gates': [dict(gate(), severity='invalid')]}, 'ValueError'),
        ]:
            result = outcome(local_scoring._evaluate_gates_and_scores, evaluation, root, root, {})
            assert result[:2] == ('error', error)
    forbidden = ('ai4sci_bench.generators', 'ai4sci_bench.adapters',
                 'ai4sci_bench.scorers.llm_judge')
    assert not any(name.startswith(forbidden) for name in sys.modules)
    print(f'{group}: passed')


if __name__ == '__main__':
    run_contract(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))


def test_review_single_metadata_load_and_native_gate_status():
    """Guards review items 2/5: no output projection or duplicate scoring state."""
    functions = {node.name: node for node in ast.parse(BRIDGE.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    assert '_merge_output_files' not in functions and 'merge_task_eval' not in functions
    def calls(name, target):
        return [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == target]
    assert len(calls('evaluate_request', 'load_scoring_metadata')) == 1
    assert 'preflight_task_environment' not in functions
    assert len(calls('register_task_scorers', 'check_evaluator_imports')) == 2
    assert not calls('build_evaluation_report', 'all')


def test_generic_evaluate_request_forwards_request_context(tmp_path):
    """Guards generic entry against fixed task/b1 calls in baseline commit 9b90d16c."""
    tree = ast.parse(BRIDGE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'evaluate_request')
    launcher = tmp_path / 'opt/bridge/test.sh'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('# test launcher')
    inputs = tmp_path / 'input'
    inputs.mkdir()
    (inputs / 'instance_parameters.json').write_text('{"n": 7}')
    (inputs / 'evaluation_manifest.json').write_text('{}')
    metadata = {'id': 'text.example', 'evaluation': {'scoring': [{'scorer': 'answer', 'weight': 10}]}}
    request = {'task_id': 'text.example', 'prompt_level': 'b2', 'files': {}, 'runtime': {
        'bridge_sha256': hashlib.sha256(BRIDGE.read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(),
        'evaluator_bundle_digest': 'a' * 64,
    }}
    calls = []

    def register(task_dir, task_id, evaluation, evaluator_root):
        calls.append(('register', task_id))
        return {}

    def score(evaluation, pred_dir, ref_dir, parameters, *, prompt_level):
        calls.append(('score', prompt_level))
        assert pred_dir == Path('/prediction') and ref_dir == inputs / 'instance/reference'
        assert parameters == {'n': 7}
        return [], True, 0, [], 7

    namespace = dict(
        Path=lambda path: tmp_path / str(path).lstrip('/') if str(path).startswith('/opt/') else Path(path),
        __file__=str(BRIDGE), hashlib=hashlib, json=json,
        _load_regular_json=lambda path, **kwargs: {
            'bundle_digest': 'a' * 64, 'launcher': {'sha256': request['runtime']['launcher_sha256']}},
        verify_scoring_module=lambda manifest: None,
        load_scoring_metadata=lambda path: metadata,
        register_task_scorers=register,
        _evaluate_gates_and_scores=score,
        build_evaluation_report=lambda req, gates, scores, value, maximum: {
            'score': value, 'max_score': maximum, 'provenance': {}},
        _input_inventory=lambda path: {},
    )
    # Compile only the orchestration function; all I/O boundaries are local fixtures.
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BRIDGE), 'exec'), namespace)
    report = namespace['evaluate_request'](request, inputs)
    assert calls == [('register', 'text.example'), ('score', 'b2')]
    assert report['score'] == 7 and report['max_score'] == 10
