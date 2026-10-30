"""Service credentials never go to the agent, argv, provenance or retained files."""
import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest
from ai4sci_bench.runner.os_sandbox import OSSandbox


def metadata(**kwargs):
    return {'runtime': {'services': [{'name': 'wolfram', 'image': 'local/image:fixed', 'port': 8765, **kwargs}]}}


@pytest.mark.parametrize('extra', [
    {'env_from': {'LD_PRELOAD': 'TOKEN'}}, {'env_from': {'HOME': 'TOKEN'}},
    {'env_from': {'TOKEN': 'not an env name'}}, {'env_from': {'ASIBENCH_SERVICE_BAD': 'TOKEN'}},
    {'environment': {'TOKEN': 'plain'}, 'env_from': {'TOKEN': 'TOKEN'}},
    {'memory_mb': True}, {'memory_mb': 128}, {'memory_mb': 65537},
    {'cpus': True}, {'cpus': float('nan')}, {'cpus': .1}, {'cpus': 17},
])
def test_invalid_service_options(extra):
    with pytest.raises(ValueError): OSSandbox._task_services(metadata(**extra))


def test_secret_env_file_scoped_and_deleted(monkeypatch, tmp_path):
    monkeypatch.setenv('WOLFRAM_TEST_LICENSE', 'private-test-value-not-a-real-license')
    services=OSSandbox._task_services(metadata(env_from={'WOLFRAM_LICENSE_B64':'WOLFRAM_TEST_LICENSE'},memory_mb=8192,cpus=2))
    sandbox=OSSandbox(tmp_path,image_builder=MagicMock());sandbox._wait_for_task_service=MagicMock()
    observed={}
    def run(cmd,**kwargs):
        assert 'private-test-value' not in repr(cmd)
        path=Path(cmd[cmd.index('--env-file')+1]);observed['path']=path
        assert path.read_text()=='WOLFRAM_LICENSE_B64=private-test-value-not-a-real-license\n'
        assert path.stat().st_mode & 0o777==0o600
        assert cmd[cmd.index('--memory')+1]=='8192m'
        assert cmd[cmd.index('--cpus')+1]=='2.0'
        assert not any(x in cmd for x in ['-p','--publish','--network=host'])
        return subprocess.CompletedProcess(cmd,0,'container-id','')
    monkeypatch.setattr(subprocess,'run',run)
    containers=[];env=sandbox._start_task_services(services,'private-network',containers)
    assert len(containers)==1 and list(env)==['ASIBENCH_SERVICE_WOLFRAM_URL']
    assert 'private-test-value' not in repr(env)
    assert not observed['path'].exists()


@pytest.mark.parametrize('value',[None,'multiline\nsecret','carriage\rsecret','nul\0secret'])
def test_missing_or_multiline_credentials_fail_closed(value,monkeypatch,tmp_path):
    if value is None: monkeypatch.delenv('LICENSE_INPUT',raising=False)
    else: monkeypatch.setenv('LICENSE_INPUT',value) if '\0' not in value else None
    if value is not None and '\0' in value:
        # os.environ itself refuses NUL; exercise the runner via a mapping fixture.
        monkeypatch.setattr('ai4sci_bench.runner.os_sandbox.os.environ',{'LICENSE_INPUT':value})
    sandbox=OSSandbox(tmp_path,image_builder=MagicMock())
    run=MagicMock();monkeypatch.setattr(subprocess,'run',run)
    services=OSSandbox._task_services(metadata(env_from={'LICENSE':'LICENSE_INPUT'}))
    with pytest.raises(ValueError):sandbox._start_task_services(services,'private',[])
    run.assert_not_called()


def test_docker_failure_does_not_echo_secret(monkeypatch,tmp_path):
    monkeypatch.setenv('LICENSE_INPUT','hidden-fixture')
    sandbox=OSSandbox(tmp_path,image_builder=MagicMock())
    paths=[]
    def run(cmd,**kwargs):
        paths.append(Path(cmd[cmd.index('--env-file')+1]))
        return subprocess.CompletedProcess(cmd,1,'','error includes hidden-fixture')
    monkeypatch.setattr(subprocess,'run',run)
    services=OSSandbox._task_services(metadata(env_from={'LICENSE':'LICENSE_INPUT'}))
    with pytest.raises(RuntimeError) as exc:sandbox._start_task_services(services,'private',[])
    assert 'hidden-fixture' not in str(exc.value) and not paths[0].exists()


def test_timed_out_docker_start_registers_container_for_cleanup(monkeypatch,tmp_path):
    monkeypatch.setenv('LICENSE_INPUT','hidden-fixture')
    sandbox=OSSandbox(tmp_path,image_builder=MagicMock());paths=[]
    def run(cmd,**kwargs):
        paths.append(Path(cmd[cmd.index('--env-file')+1]))
        raise subprocess.TimeoutExpired(cmd,60)
    monkeypatch.setattr(subprocess,'run',run)
    services=OSSandbox._task_services(metadata(env_from={'LICENSE':'LICENSE_INPUT'}))
    containers=[]
    with pytest.raises(subprocess.TimeoutExpired):sandbox._start_task_services(services,'private',containers)
    assert len(containers)==1 and containers[0].startswith('ai4sci-svc-')
    assert not paths[0].exists()
