"""Accepted tasks must grade without private source and distinguish runner failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    path=ROOT/'tasks/computer_science'/name/'custom_scorer.py'
    spec=importlib.util.spec_from_file_location('accepted_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_qp_never_reads_generator_or_reference_solver(tmp_path,monkeypatch):
    m=load('qp_sensitivity_audit');pred=tmp_path/'pred';ref=tmp_path/'reference';pred.mkdir();ref.mkdir()
    (pred/'solver.py').write_text('pass');(ref/'private_atlas_cases.npz').write_bytes(b'problems')
    # A legacy generator stand-in cannot activate a bypass in a public scorer.
    (ref/'reference_solver.py').write_text('private implementation')
    monkeypatch.setattr(m,'run_isolated',lambda *a:(b'output',{'sandbox':'docker-os'}))
    assert m.execute(pred,ref,{})[0]==b'output'

@pytest.mark.parametrize('name,cls,source,reference',[
    ('qp_sensitivity_audit','QPActivePathScorer','solver.py','private_atlas_answers.npz'),
    ('minimal_latent_realization','LatentScorer','factorizer.py','evaluation_inputs.json'),
    ('signed_ising_log_partition','SignedIsingScorer','normalizer.py','evaluation_inputs.json'),
])
def test_missing_reference_is_evaluator_failure(tmp_path,name,cls,source,reference):
    m=load(name);pred=tmp_path/'pred';ref=tmp_path/'ref';pred.mkdir();ref.mkdir();(pred/source).write_text('pass')
    result=getattr(m,cls)().score(pred,ref,{})
    assert result.details.get('scorer_internal_error') is True
    assert result.details['failure_kind']=='missing_evaluator_input'

@pytest.mark.parametrize('name,cls,source',[
    ('minimal_latent_realization','LatentScorer','factorizer.py'),
    ('signed_ising_log_partition','SignedIsingScorer','normalizer.py'),
])
@pytest.mark.parametrize('infra',['Docker daemon unavailable',None])
def test_runner_failure_classification(tmp_path,monkeypatch,name,cls,source,infra):
    m=load(name);pred=tmp_path/'pred';ref=tmp_path/'ref';pred.mkdir();ref.mkdir();(pred/source).write_text('raise RuntimeError()')
    (ref/'evaluation_inputs.json').write_text('[]')
    monkeypatch.setattr(m,'SandboxedSubmissionRunner',lambda *a:SimpleNamespace(run=lambda spec:SimpleNamespace(infrastructure_error=infra,timed_out=False,exit_code=1,stderr_tail='submission failed',image_identity='sha256:test',output_bytes=lambda n:None)))
    result=getattr(m,cls)().score(pred,ref,{})
    assert bool(result.details.get('scorer_internal_error'))==bool(infra)
    if infra:assert result.details['failure_kind']=='evaluator_unavailable'

@pytest.mark.parametrize('corrupt',[False,True])
def test_click_reference_failure_is_not_submission_zero(tmp_path,corrupt):
    m=load('heterogeneous_click_log_counterfactuals');pred=tmp_path/'pred';ref=tmp_path/'ref';pred.mkdir();ref.mkdir()
    if corrupt:(ref/'counterfactual_ctr_ref.csv').write_text('wrong,header\n1,2\n')
    result=m.HeterogeneousClickLogScorer().score(pred,ref,{})
    assert result.details['scorer_internal_error'] is True
    assert result.details['failure_kind']==('evaluator_runtime_error' if corrupt else 'missing_evaluator_input')
