"""Run only submitted factorizer.py and public inputs in an offline sandbox."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.task import TaskLoader as SDKTaskLoader
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.runner.submission_sandbox import ReadOnlyMount, SandboxedSubmissionRunner, SubmissionRunSpec

TASK = Path(__file__).resolve().parent
RUNNER = '''import importlib.util,json,numbers,sys
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location("submission","/workspace/factorizer.py")
module=importlib.util.module_from_spec(spec);sys.modules['submission']=module;spec.loader.exec_module(module)
cases=json.loads(Path('/workspace/data/evaluation_inputs.json').read_text())
answers=[]
for case in cases:
    try:
        result=module.factorize(case)
    except Exception as exc:
        answers.append({'id':case['id'],'submission_error':type(exc).__name__})
        continue
    try:
        if not isinstance(result,dict):raise ValueError('dictionary required')
        k=result.get('K')
        if isinstance(k,bool) or not isinstance(k,numbers.Integral) or k<=0:
            raise ValueError('positive integer K required')
        answer={'id':case['id'],'K':int(k)}
        for key in ('W','H'):
            raw=np.asarray(result[key])
            if raw.ndim!=2:raise ValueError('matrix required')
            if any(isinstance(x,(str,bytes,bool,complex)) for row in raw.tolist() for x in row):
                raise ValueError('real numeric entries required')
            answer[key]=np.asarray(raw,dtype=float).tolist()
        answers.append(answer)
    except Exception as exc:
        answers.append({'id':case['id'],'invalid_output':type(exc).__name__})
Path('/workspace/predictions.json').write_text(json.dumps(answers,allow_nan=True))
'''


def score_case(case, reference, answer):
    """The same public tolerances apply at every seed and prompt level."""
    record = {'id': reference['id'], 'expected_K': reference['minimum_K'],
              'returned_K': None, 'K_correct': False, 'factors_valid': False, 'factors_finite': False,
              'max_entry_product_error': None, 'W_column_sum_error': None,
              'H_column_sum_error': None, 'minimum_factor_entry': None,
              'credit': 0.0}
    if not isinstance(answer, dict):
        record['invalid_output'] = 'missing answer'
        return record
    if 'submission_error' in answer or 'invalid_output' in answer:
        record.update({k: answer[k] for k in ('submission_error', 'invalid_output') if k in answer})
        return record
    k = answer.get('K')
    record['returned_K'] = k if isinstance(k, int) and not isinstance(k, bool) else None
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        record['invalid_output'] = 'positive integer K required'
        return record
    record['K_correct'] = k == reference['minimum_K']
    try:
        m = np.array([[float(Fraction(x)) for x in row] for row in case['M']])
        w, h = np.asarray(answer.get('W'), dtype=float), np.asarray(answer.get('H'), dtype=float)
        if w.shape != (len(m), k) or h.shape != (k, m.shape[1]):
            raise ValueError('wrong shapes')
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(h)):
            raise ValueError('nonfinite factors')
        record['factors_finite'] = True
        minimum = float(min(np.min(w), np.min(h)))
        record['minimum_factor_entry'] = minimum
        if minimum < -1e-10:
            raise ValueError('negative factors beyond roundoff allowance')
        # Public contract: clamp permitted negative roundoff before all checks.
        w, h = np.maximum(w, 0), np.maximum(h, 0)
        we = float(np.max(np.abs(w.sum(axis=0) - 1)))
        he = float(np.max(np.abs(h.sum(axis=0) - 1)))
        residual = float(np.max(np.abs(w @ h - m)))
        record.update(W_column_sum_error=we, H_column_sum_error=he,
                      max_entry_product_error=residual)
        if not np.isfinite(residual) or we > 1e-6 or he > 1e-6:
            raise ValueError('nonstochastic factors or nonfinite product')
        record['factors_valid'] = True
        if record['K_correct']:
            record['credit'] = max(0.0, min(1.0, (1e-3 - residual) / (1e-3 - 1e-6)))
    except (ValueError, TypeError, OverflowError):
        record['invalid_output'] = 'invalid factors'
    return record


@register_scorer('minimal_latent_realization_v1')
class LatentScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get('weight', 100))
        source = pred_dir / 'factorizer.py'
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(pred_dir.resolve()):
            return ScoreDetail(self.name, 0, weight, False, {'execution_ok': False}, 'Missing regular factorizer.py')
        metadata = SDKTaskLoader(TASK.parents[1]).load_task_by_id('computer_science.minimal_latent_realization')
        with tempfile.TemporaryDirectory(prefix='asi_latent_grade_') as tmp:
            workspace = Path(tmp)
            (workspace / 'data').mkdir()
            shutil.copyfile(source, workspace / 'factorizer.py')
            (workspace / 'evaluate.py').write_text(RUNNER)
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
            try:
                predictions = json.loads(output.read_text())
                if not isinstance(predictions, list): raise ValueError('list required')
                by_id = {item['id']: item for item in predictions}
                if len(by_id) != len(predictions): raise ValueError('duplicate ids')
            except (ValueError, TypeError, KeyError):
                return ScoreDetail(self.name, 0, weight, False, {'execution_ok': False}, 'Malformed evaluator output')
        inputs = json.loads((ref_dir / 'evaluation_inputs.json').read_text())
        refs = json.loads((ref_dir / 'latent_ref.json').read_text())['cases']
        cases = {case['id']: case for case in inputs}
        details = [score_case(cases[ref['id']], ref, by_id.get(ref['id'])) for ref in refs]
        execution_ok = len(predictions) == len(refs)
        score = weight * sum(record['credit'] for record in details) / len(details)
        return ScoreDetail(self.name, score, weight, execution_ok and all(r['credit'] == 1 for r in details),
            {'cases': details, 'execution_ok': execution_ok, 'image': image},
            f'{sum(r["credit"] == 1 for r in details)}/{len(details)} joint K/factor results fully correct')


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

LatentScorer.score = _classify_evaluator_failure(LatentScorer.score)
