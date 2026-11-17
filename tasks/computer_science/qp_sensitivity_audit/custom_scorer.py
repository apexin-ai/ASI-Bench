"""Hidden-family scorer for affine-parametric QP continuation paths."""
from __future__ import annotations
import hashlib,io,math,os,subprocess,sys,tempfile,threading
from concurrent.futures import Future
from pathlib import Path
import numpy as np
from ai4sci_bench.core.scorer import Scorer,register_scorer
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.core.types import ScoreDetail
try:
 from ai4sci_bench.runner.submission_sandbox import ReadOnlyMount,SandboxedSubmissionRunner,SubmissionRunSpec
except ImportError: ReadOnlyMount=SandboxedSubmissionRunner=SubmissionRunSpec=None
TASK_DIR=Path(__file__).resolve().parent;REPO_ROOT=TASK_DIR.parents[2];FAMILIES=("regular_dense","scaled_redundancy","simultaneous_exchange","ill_conditioned");_LOCK=threading.Lock();_CACHE={}
REQUIRED=("region_count","vertex_count","vertices","x_intercept","x_theta","tight_mask","adjacency")
def frac(error,full,zero):
 if not math.isfinite(error) or error>=zero:return 0.
 if error<=full:return 1.
 return float((zero-error)/(zero-full))
def digest(root):
 h=hashlib.sha256()
 for p in sorted(root.rglob("*")):
  if p.is_symlink():raise ValueError("submission symlink")
  if p.is_file():h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
 return h.hexdigest()
def run_isolated(pred,problem,timeout):
 if SandboxedSubmissionRunner is None:raise RuntimeError("Docker submission API unavailable")
 meta=TaskLoader(REPO_ROOT/"tasks").load_task_by_id("computer_science.qp_sensitivity_audit")
 spec=SubmissionRunSpec(task_metadata=meta,submission_dir=pred,command=("/opt/venv/bin/python","/submission/solver.py","--input","/input/problems.npz","--output","/output/atlas.npz"),readonly_inputs=(ReadOnlyMount(problem,"/input/problems.npz"),),output_files=("atlas.npz",),timeout_seconds=timeout,cpu_limit=2.,memory_limit="3g",pids_limit=512,max_output_file_bytes=64*1024*1024)
 r=SandboxedSubmissionRunner(REPO_ROOT).run(spec)
 if r.infrastructure_error:raise RuntimeError(r.infrastructure_error)
 if r.timed_out:raise TimeoutError(f"solver exceeded {timeout}s")
 if r.exit_code!=0:raise ValueError(r.stderr_tail[-2000:])
 payload=r.output_bytes("atlas.npz")
 if not payload:raise ValueError("missing output")
 return payload,{"elapsed_seconds":r.elapsed_seconds,"sandbox":"docker-os","image_identity":r.image_identity}
def execute(pred,ref,config):
 problem=ref/config.get("private_problem_file","private_atlas_cases.npz");timeout=float(config.get("runtime_timeout_seconds",180));key=hashlib.sha256((digest(pred)+hashlib.sha256(problem.read_bytes()).hexdigest()+str(timeout)).encode()).hexdigest()
 with _LOCK:
  f=_CACHE.get(key);owner=f is None
  if f is None:f=Future();_CACHE[key]=f
 if owner:
  try:
   value=run_isolated(pred,problem,timeout)
   f.set_result(value)
  except BaseException as e:f.set_exception(e)
 return f.result()
def load_output(payload,ref):
 with np.load(io.BytesIO(payload),allow_pickle=False) as p:
  if set(p.files)!=set(REQUIRED):raise ValueError("invalid output keys")
  out={k:p[k] for k in p.files}
 for k in REQUIRED:
  expected=ref[k].shape
  if out[k].shape!=expected:raise ValueError(f"{k} shape {out[k].shape}, expected {expected}")
 if out["region_count"].dtype!=np.int64 or out["vertex_count"].dtype!=np.int64 or out["tight_mask"].dtype!=np.bool_ or out["adjacency"].dtype!=np.bool_:raise ValueError("integer/bool dtype contract failed")
 for k in ("vertices","x_intercept","x_theta"):
  if out[k].dtype!=np.float64 or not np.isfinite(out[k]).all():raise ValueError(f"{k} invalid")
 return out
def harmonic(values):return 0. if not values or min(values)<=0 else len(values)/sum(1/v for v in values)
@register_scorer("qp_active_path_v3")
class QPActivePathScorer(Scorer):
 def score(self,pred_dir:Path,ref_dir:Path,config:dict):
  weight=float(config.get("weight",100.))
  try:
   if not (pred_dir/"solver.py").is_file():return ScoreDetail("qp_active_path_v3",0,weight,False,{},"Missing solver.py")
   try:
    with np.load(ref_dir/config.get("private_answer_file","private_atlas_answers.npz"),allow_pickle=False) as z:ref={k:z[k] for k in z.files}
   except (ValueError,KeyError,EOFError) as e:return ScoreDetail("qp_active_path_v3",0,weight,False,{"scorer_internal_error":True,"failure_kind":"evaluator_runtime_error","error":repr(e)},"Invalid reference bundle")
   payload,runtime=execute(pred_dir,ref_dir,config);p=load_output(payload,ref);family=[];details={}
   for fid,name in enumerate(FAMILIES):
    case_scores=[]
    for i in np.flatnonzero(ref["family_index"]==fid):
     r=int(ref["region_count"][i]);rp=int(np.clip(p["region_count"][i],0,ref["x_intercept"].shape[1]));count=float(rp==r)
     if rp==r and r>0:
      vc=float(np.mean(p["vertex_count"][i,:r]==ref["vertex_count"][i,:r]));pe=[]
      for j in range(r):
       v=int(ref["vertex_count"][i,j]);pe.append(float(np.max(abs(p["vertices"][i,j,:v]-ref["vertices"][i,j,:v]))) if v else 0.)
      polygon=frac(max(pe),1e-7,2e-3);xe=max(np.linalg.norm(p["x_intercept"][i,:r]-ref["x_intercept"][i,:r])/max(1,np.linalg.norm(ref["x_intercept"][i,:r])),np.linalg.norm(p["x_theta"][i,:r]-ref["x_theta"][i,:r])/max(1,np.linalg.norm(ref["x_theta"][i,:r])));tight=float(np.mean(p["tight_mask"][i,:r]==ref["tight_mask"][i,:r]));adj=float(np.mean(p["adjacency"][i,:r,:r]==ref["adjacency"][i,:r,:r]))
     else:vc=polygon=tight=adj=0.;xe=float("inf")
     q=.20*count+.20*vc+.20*polygon+.25*frac(xe,1e-7,5e-2)+.10*tight+.05*adj;case_scores.append(q)
    f=harmonic(case_scores);family.append(f);details[name]={"score_fraction":f,"case_fractions":case_scores}
   mean=float(np.mean(family));worst=float(min(family));h=harmonic(family);score=100*(.3*mean+.4*worst+.3*h)
   return ScoreDetail("qp_active_path_v3",score*weight/100,weight,score>=60,{"score_100":score,"families":details,"mean_family_fraction":mean,"worst_family_fraction":worst,"runtime":runtime},f"active-path score={score:.2f}/100")
  except (TimeoutError,subprocess.TimeoutExpired):return ScoreDetail("qp_active_path_v3",0,weight,False,{"error":"timeout"},"solver timeout")
  except (FileNotFoundError,PermissionError) as e:return ScoreDetail("qp_active_path_v3",0,weight,False,{"scorer_internal_error":True,"failure_kind":"missing_evaluator_input","error":repr(e)},"Missing evaluator input")
  except RuntimeError as e:return ScoreDetail("qp_active_path_v3",0,weight,False,{"scorer_internal_error":True,"failure_kind":"evaluator_unavailable","error":repr(e)},"Evaluator unavailable")
  except Exception as e:return ScoreDetail("qp_active_path_v3",0,weight,False,{"error":repr(e)},f"evaluation failed: {e}")
