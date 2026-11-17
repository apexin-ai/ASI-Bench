"""Score independently executed graph normalizers with references outside Docker."""
from pathlib import Path
import json
import math
import shutil
import tempfile

from ai4sci_bench.core.scorer import Scorer,register_scorer
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.runner.submission_sandbox import ReadOnlyMount, SandboxedSubmissionRunner, SubmissionRunSpec

TASK=Path(__file__).resolve().parent
RUNNER='''import importlib.util,json,math,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("submission","/workspace/normalizer.py")
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
cases=json.loads(Path('/workspace/data/evaluation_inputs.json').read_text())
answers=[]
for case in cases:
    try:
        value=module.log_partition(case['graph'])
        if isinstance(value,bool):raise ValueError('boolean result')
        value=float(value)
        if not math.isfinite(value):raise ValueError('nonfinite result')
        answers.append({'id':case['id'],'log_partition':value})
    except Exception as exc:
        answers.append({'id':case['id'],'error':type(exc).__name__})
Path('/workspace/predictions.json').write_text(json.dumps(answers,allow_nan=False))
'''


@register_scorer('signed_ising_log_partition_v3')
class SignedIsingScorer(Scorer):
    def score(self,pred_dir:Path,ref_dir:Path,config:dict)->ScoreDetail:
        weight=float(config.get('weight',100));source=pred_dir/'normalizer.py'
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(pred_dir.resolve()):
            return ScoreDetail(self.name,0,weight,False,{},'Missing normalizer.py')
        metadata=TaskLoader(TASK.parents[1]).load_task_by_id('computer_science.signed_ising_log_partition')
        with tempfile.TemporaryDirectory(prefix='asi_signed_ising_grade_') as tmp:
            workspace=Path(tmp);(workspace/'data').mkdir()
            shutil.copyfile(source,workspace/'normalizer.py')
            (workspace/'evaluate.py').write_text(RUNNER)
            runner_source = RUNNER.replace('/workspace/data/evaluation_inputs.json', '/input/evaluation_inputs.json').replace('/workspace/predictions.json', '/output/predictions.json').replace('/workspace/', '/submission/')
            (workspace / 'evaluate.py').write_text(runner_source)
            result = SandboxedSubmissionRunner(TASK.parents[2]).run(SubmissionRunSpec(
                task_metadata=metadata, submission_dir=workspace,
                command=('/opt/venv/bin/python','/submission/evaluate.py'),
                readonly_inputs=(ReadOnlyMount(ref_dir/'evaluation_inputs.json','/input/evaluation_inputs.json'),),
                output_files=('predictions.json',), timeout_seconds=600, cpu_limit=4.0, memory_limit='8g'))
            image = result.image_identity
            if result.infrastructure_error:
                return ScoreDetail(self.name,0,weight,False,{'scorer_internal_error':True,'failure_kind':'evaluator_unavailable','image':image,'error':result.infrastructure_error},'Evaluator unavailable')
            payload = result.output_bytes('predictions.json')
            if result.timed_out or result.exit_code != 0 or payload is None:
                return ScoreDetail(self.name,0,weight,False,{'execution_ok':False,'image':image,'log_tail':result.stderr_tail},'Submission execution failed')
            output = workspace/'predictions.json'
            output.write_bytes(payload)
            try:by_id={x['id']:x.get('log_partition') for x in json.loads(output.read_text())}
            except (ValueError,TypeError,KeyError):
                return ScoreDetail(self.name,0,weight,False,{'execution_ok':True},'Malformed returned values')
        reference=json.loads((ref_dir/'partition_ref.json').read_text());details=[]
        for case in reference:
            value=by_id.get(case['id']);error=None;credit=0.0
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                error=abs(value-case['log_partition'])
                credit=max(0.,min(1.,(1e-3-error)/(1e-3-1e-6)))
            details.append({'id':case['id'],'absolute_log_error':error,'credit':credit})
        score=weight*sum(x['credit'] for x in details)/len(details)
        return ScoreDetail(self.name,score,weight,all(x['credit']==1 for x in details),
                           {'cases':details,'execution_ok':True,'image':image},
                           f'{sum(x["credit"]==1 for x in details)}/{len(details)} normalizers fully correct')


def _classify_evaluator_failure(score):
    def wrapped(self, pred_dir, ref_dir, config):
        try:
            if not (Path(ref_dir)/'evaluation_inputs.json').is_file():
                raise FileNotFoundError('Missing evaluation_inputs.json')
            return score(self, pred_dir, ref_dir, config)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            kind = 'missing_evaluator_input' if isinstance(exc, OSError) else 'evaluator_runtime_error'
            return ScoreDetail(self.name,0,float(config.get('weight',100)),False,
                {'scorer_internal_error':True,'failure_kind':kind,'error':str(exc)},'Invalid evaluator input')
    return wrapped

SignedIsingScorer.score = _classify_evaluator_failure(SignedIsingScorer.score)
