"""Checkpoint integrity and opt-in offline deployment acceptance for CHGNet."""
import hashlib
import io
import json
import os
import subprocess

import pytest

from ai4sci_bench.mcp_config import load_mcp_config
from .support import ROOT, client, load
from scripts.mcp.e2e.e2e_smoke.materials import REFERENCE, SI2, payload

DEPLOY = ROOT / 'scripts/mcp/materials'
MODEL = '/opt/models/CHGNet-PES-MatPES-PBE-1M-2026.9'


@pytest.mark.parametrize('corrupt', [False, True])
def test_checkpoint_download_enforces_hash_before_writing(tmp_path, monkeypatch, corrupt):
    module = load(DEPLOY / 'fetch_model.py', 'chgnet_fetch_model')
    data = b'checkpoint'
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps(dict(repository='owner/model', revision='a'*40,
        files={'state.pt': hashlib.sha256(data).hexdigest()})))
    urls = []
    def download(url, timeout):
        urls.append(url)
        return io.BytesIO(b'corrupt' if corrupt else data)
    monkeypatch.setattr(module.urllib.request, 'urlopen', download)
    output = tmp_path / 'model'
    if corrupt:
        with pytest.raises(ValueError, match='SHA-256 mismatch'):
            module.fetch(lock, output)
        assert not (output / 'state.pt').exists()
    else:
        module.fetch(lock, output)
        assert (output / 'state.pt').read_bytes() == data
    assert urls == ['https://huggingface.co/owner/model/resolve/' + 'a'*40 + '/state.pt']


def test_chgnet_config_can_be_staged_for_os_codex():
    from ai4sci_bench.runner.mcp_sandbox import stage_os_mcp
    servers = load_mcp_config(DEPLOY / 'mcp.chgnet.os.json')
    with stage_os_mcp(servers, {}, 'codex', ['codex', 'exec'], None) as (cmd, _, _):
        assert 'mcp_servers.chgnet.required=true' in cmd


@pytest.mark.skipif(os.environ.get('ASIBENCH_CHGNET_DOCKER') != '1', reason='requires built CHGNet image')
def test_offline_image_mcp_prediction_matches_independent_backend(tmp_path):
    image = os.environ.get('ASIBENCH_CHGNET_IMAGE', 'asibench-chgnet:20261031')
    server = json.loads((DEPLOY / 'mcp.chgnet.os.json').read_text())['mcpServers']['chgnet']
    # A fresh HOME models the framework's isolated agent home. No host cache,
    # science venv, credentials or network is available to this container.
    args = ['run', '--rm', '-i', '--network', 'none', '--cpus', '2', '--memory', '4g',
            '-e', 'HOME=/tmp/fresh-home', '--entrypoint', server['command']]
    for key, value in server['env'].items():
        args += ['-e', f'{key}={value}']
    args += [image, *server['args']]
    mcp = client.StdioMCP('docker', args, cwd=tmp_path, env=dict(os.environ), stderr_path=tmp_path/'stderr.log')
    try:
        mcp.initialize()
        assert {t['name'] for t in mcp.list_tools()} == {
            'get_info', 'load_model', 'predict_structure', 'predict_atomic_features',
            'predict_bandgap', 'relax_structure', 'run_md'}
        loaded = payload(mcp.call_tool('load_model', {'model_name': MODEL, 'device': 'cpu'})['result'])
        assert loaded.startswith('Successfully loaded '), loaded
        actual = payload(mcp.call_tool('predict_structure', {'structure_data': SI2})['result'])
        assert not actual.get('error'), actual
    finally:
        mcp.close()
    assert not mcp.non_json_stdout
    # Separate process calls the official ASE backend, without the MCP wrapper.
    reference_code = ('import tempfile\nfrom pathlib import Path\n'
        'd=Path(tempfile.mkdtemp())\n'
        f'(d/"in.json").write_text({json.dumps(SI2)!r})\n'
        f'import sys\nsys.argv=["ref", "chgnet", {MODEL!r}, str(d/"in.json"), str(d/"out.json")]\n'
        + REFERENCE + '\nprint((d/"out.json").read_text())\n')
    reference = subprocess.run(['docker', 'run', '--rm', '--network', 'none',
        '--cpus', '2', '--memory', '4g', '--entrypoint', server['command'], image, '-c', reference_code],
        capture_output=True, text=True, check=True, timeout=300)
    expected = json.loads(reference.stdout.splitlines()[-1])
    (tmp_path/'prediction.json').write_text(json.dumps(dict(actual=actual, reference=expected), indent=2))
    def flat(value):
        return [x for row in value for x in flat(row)] if isinstance(value, list) else [value]
    for key in ('energy', 'forces', 'stress'):
        assert flat(actual[key]) == pytest.approx(flat(expected[key]), abs=1e-5, rel=0)
