"""The public scorer must consume materialized replays without a GT generator."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_scorer():
    path = ROOT / 'tasks/finance/robust_arbitrage_free_option_surface/custom_scorer.py'
    spec = importlib.util.spec_from_file_location('accepted_finance_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_replay_is_evaluator_failure(tmp_path):
    module = load_scorer()
    (tmp_path / 'analysis.py').write_text('pass\n')
    detail = module.OptionSurfaceHiddenReplayGeneralizationScorer().score(
        tmp_path, tmp_path / 'reference', {'weight': 65})
    assert detail.details['scorer_internal_error'] is True
    assert detail.details['failure_kind'] == 'missing_evaluator_input'


def test_materialized_replay_only_exposes_data_to_program(tmp_path, monkeypatch):
    module = load_scorer()
    pred = tmp_path / 'prediction'
    ref = tmp_path / 'reference'
    pred.mkdir()
    (pred / 'analysis.py').write_text('pass\n')
    bundle = ref / 'replay/replay_1'
    (bundle / 'data').mkdir(parents=True)
    (bundle / 'reference').mkdir()
    (bundle / 'data/input.json').write_text('{"test": 1}')
    (bundle / 'reference/answer.json').write_text('{"private": 1}')
    observed = []

    def fake_run(*args, **kwargs):
        workspace = kwargs['cwd']
        observed.append(workspace)
        assert json.loads((workspace / 'data/input.json').read_text()) == {'test': 1}
        assert (workspace / 'analysis.py').read_text() == 'pass\n'
        assert not (workspace / 'reference').exists()
        assert not (workspace / 'generate_gt.py').exists()
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    monkeypatch.setattr(module, '_hidden_component_fraction', lambda *args: (1.0, {}))
    detail = module.OptionSurfaceHiddenReplayGeneralizationScorer().score(
        pred, ref, {'weight': 65, 'replay_ids': ['replay_1']})
    assert observed
    assert detail.score == 65
