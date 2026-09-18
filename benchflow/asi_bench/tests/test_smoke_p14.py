"""Guards P1.4 frozen-prediction binding and evaluation status classification."""
import json

import pytest

from benchmarks.asi_bench.tests.smoke_p14 import AUDIT, freeze_prediction


def test_freeze_and_tamper(tmp_path):
    import hashlib
    result = tmp_path / 'result.json'
    outputs = tmp_path / 'prediction'
    outputs.mkdir()
    entries = []
    for name in ('analysis.py', 'certificate.json'):
        data = b'{}'
        (outputs / name).write_bytes(data)
        entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    value = {'attempt_status': 'completed', 'prediction_status': 'present',
             'agent_output': {'persisted_outputs': {'dir': 'prediction', 'files': entries}}}
    result.write_text(json.dumps(value))
    freeze_prediction(result, tmp_path / 'frozen')
    assert (tmp_path / 'frozen/analysis.py').read_bytes() == b'{}'
    (outputs / 'analysis.py').write_text('tamper')
    with pytest.raises(ValueError, match='integrity'):
        freeze_prediction(result, tmp_path / 'bad')


@pytest.mark.parametrize('path', ['../prediction', '/tmp/prediction'])
def test_reject_escaping_prediction(tmp_path, path):
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'attempt_status': 'completed', 'prediction_status': 'present',
        'agent_output': {'persisted_outputs': {'dir': path, 'files': []}}}))
    with pytest.raises(ValueError, match='path'):
        freeze_prediction(result, tmp_path / 'frozen')


def test_p15_observability_does_not_implement_scoring():
    compile(AUDIT, '<audit>', 'exec')
    assert '_evaluate_gates_and_scores' not in AUDIT
    assert 'scorer.score' not in AUDIT
    assert "'_build_env', '_run', '_acquire_lock'" in AUDIT


def test_launcher_uses_image_interpreter():
    """Guards the image-interpreter replacement of baseline commit 9b90d16c."""
    import subprocess

    from benchmarks.asi_bench.scorer import ADAPTER_ROOT
    launcher = ADAPTER_ROOT / 'templates/verifier/test.sh'
    subprocess.run(['/bin/sh', '-n', str(launcher)], check=True)
    assert 'exec /usr/local/bin/python /opt/bridge/score_entry.py' in launcher.read_text()
