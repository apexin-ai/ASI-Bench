"""Fixed policy-transfer scoring; no execution of submitted code."""
import json
import math
from pathlib import Path
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

WEIGHTS={'eligible_fraction':20,'success_fraction':20,'conditional_success':20,'mean_A':15,'mean_B':15}

def credit(error):
    return max(0.,min(1.,(.20-error)/.19))

def evaluate(pred_dir,ref_dir):
    pred_dir=Path(pred_dir)
    try:
        truth=json.loads((Path(ref_dir)/'predictions.json').read_text(encoding='utf-8'))
        if not truth['policies']:raise ValueError('empty reference')
    except Exception as exc:
        return {'score':None,'scorer_internal_error':True,'reason':'Reference error: '+type(exc).__name__}
    try:
        prediction=json.loads((pred_dir/'predictions.json').read_text(encoding='utf-8'))
        rows=prediction['policies'];table={r['id']:r for r in rows}
        if len(table)!=len(rows):raise ValueError('Duplicate policy id')
    except Exception as exc:
        return {'score':0.,'raw_score':0.,'core_fraction':0.,'scorer_internal_error':False,'reason':'Prediction parse error: '+type(exc).__name__}
    points={k:0. for k in WEIGHTS};errors={k:[] for k in WEIGHTS};core=[]
    for expected in truth['policies']:
        actual=table.get(expected['id'],{});good=True
        for key,weight in WEIGHTS.items():
            error=float('inf')
            try:
                value=float(actual[key]);target=float(expected[key])
                if math.isfinite(value) and value>=0 and (key not in ('eligible_fraction','success_fraction','conditional_success') or value<=1+1e-10):
                    error=abs(value-target)/max(1. if key.startswith('mean_') else 1e-12,abs(target))
            except (KeyError,TypeError,ValueError):pass
            points[key]+=weight*credit(error)/len(truth['policies'])
            errors[key].append(error if math.isfinite(error) else None)
            if key in ('eligible_fraction','success_fraction','conditional_success') and error>.01:good=False
        core.append(good)
    targets=truth['calibration_ratios'];actual=prediction.get('calibration_ratios',{})
    calibration=0.
    for key,target in targets.items():
        try:
            value=float(actual[key]);err=abs(value-target)/max(abs(target),1e-12)
            if math.isfinite(value):calibration+=credit(err)
        except (KeyError,TypeError,ValueError):pass
    points['calibration']=5*calibration/len(targets)
    for name,weight in [('analysis.py',2),('report.md',3)]:
        path=pred_dir/name;points[name]=float(path.is_file() and path.stat().st_size>0)*weight
    raw=sum(points.values());core_fraction=sum(core)/len(core)
    score=min(raw,30.) if core_fraction<.9 else raw
    return {'score':round(score,10),'raw_score':round(raw,10),'core_fraction':core_fraction,
            'scorer_internal_error':False,'points':points,'policy_errors':errors}

@register_scorer('adaptive_cohort_transport_v1')
class AdaptiveCohortTransportScorer(Scorer):
    def score(self,pred_dir,ref_dir,config):
        result=evaluate(pred_dir,ref_dir);value=result['score']
        return ScoreDetail(scorer_name=self.name,score=0. if value is None else value,max_score=100.,
                           passed=value is not None and value>=99.999,details=result,
                           message=result.get('reason','Adaptive cohort policy transfer'))
