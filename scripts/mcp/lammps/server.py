"""Repository-owned LAMMPS MCP: native input plus common atomistic workflows."""
from __future__ import annotations
import argparse
import base64
import hashlib
import os
from pathlib import Path
import signal
from mcp.server.fastmcp import FastMCP
from pydantic import StrictInt, StrictFloat
from runtime import Jobs, MAX_BYTES, safe_path, read_bytes, parse_thermo, bounded
import science

mcp = FastMCP('ASI-Bench LAMMPS', host=os.environ.get('LAMMPS_MCP_HOST','0.0.0.0'),
              port=int(os.environ.get('LAMMPS_MCP_PORT','8765')))
jobs = Jobs()


@mcp.tool()
def get_info() -> dict:
    """Probe the real engine, compiled styles/packages, units and execution limits."""
    return science.capabilities(jobs.binary)


@mcp.tool()
def get_engine_help(offset: StrictInt=0, limit: StrictInt=16000) -> dict:
    """Read paginated engine help, including compiled pair/fix/compute styles."""
    import subprocess
    bounded(offset,0,1000000,'offset'); bounded(limit,1,32000,'limit')
    process=subprocess.run([jobs.binary,'-h'],capture_output=True,text=True,timeout=20)
    if process.returncode: raise RuntimeError(process.stderr[-2000:])
    return {'text':process.stdout[offset:offset+limit], 'next_offset':offset+limit,
            'eof':offset+limit>=len(process.stdout)}


@mcp.tool()
def list_potentials(prefix: str='', offset: StrictInt=0, limit: StrictInt=100) -> dict:
    """List bundled potential files with SHA-256; suitability depends on material and pair_style."""
    bounded(offset,0,100000,'offset'); bounded(limit,1,500,'limit')
    entries=[p for p in science.list_potentials() if p['name'].startswith(prefix)]
    return {'directory':str(science.potentials_dir()), 'potentials':entries[offset:offset+limit],
            'count':len(entries), 'next_offset':min(offset+limit,len(entries)), 'eof':offset+limit>=len(entries)}


@mcp.tool()
def write_input(name: str, data: str, encoding: str='utf-8') -> dict:
    """Upload a workspace input (<=16 MiB), including binary restart via base64. No overwrite."""
    if len(data)>24*1024*1024: raise ValueError('input exceeds upload budget')
    if encoding not in {'utf-8','base64'}: raise ValueError('encoding must be utf-8 or base64')
    if Path(name).parts and Path(name).parts[0]=='lammps-runs': raise ValueError('upload to an input path, not managed job directories')
    payload=data.encode() if encoding=='utf-8' else base64.b64decode(data,validate=True)
    if len(payload)>MAX_BYTES: raise ValueError('input exceeds 16 MiB')
    path=safe_path(jobs.workspace,name); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as out: out.write(payload)
    return {'path':str(path),'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}


@mcp.tool()
def submit_script(script: str, input_files: list[str] | None=None,
                  timeout_seconds: StrictInt=120, mpi_ranks: StrictInt=1) -> dict:
    """Run native LAMMPS input asynchronously; stages relative workspace files preserving names.

    Full native input has container privileges (including shell/include commands).
    Return job_id; poll/wait, cancel, inspect files and fetch text or binary artifacts.
    """
    return jobs.submit(script,input_files,timeout_seconds,mpi_ranks)


@mcp.tool()
def list_jobs() -> dict:
    """List this workspace's recorded jobs and lifecycle state."""
    with jobs.lock: return {'jobs':[dict(jobs.records[k]) for k in sorted(jobs.records)]}


@mcp.tool()
def get_status(job_id: str) -> dict:
    """Return actual process completion/error, timeout, cancellation or running state."""
    return jobs.status(job_id)


@mcp.tool()
def wait_job(job_id: str, wait_seconds: StrictInt=30) -> dict:
    """Wait up to 60 seconds; a still-running job is not treated as completed."""
    return jobs.wait(job_id,wait_seconds)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Terminate a live job process group and record cancelled; terminal jobs are unchanged."""
    return jobs.cancel(job_id)


@mcp.tool()
def get_result(job_id: str) -> dict:
    """Get status, files, real thermo/minimizer stop reason, and high-level final atoms/forces."""
    return science.result(jobs,job_id)


@mcp.tool()
def list_files(job_id: str) -> dict:
    """List job artifacts without following symlinks; bounded to 2048 files."""
    return jobs.files(job_id)


@mcp.tool()
def read_output(job_id: str, name: str, offset: StrictInt=0, limit: StrictInt=65536,
                encoding: str='utf-8') -> dict:
    """Read paginated bytes; use base64 for binary restart/artifact transfer over HTTP."""
    return jobs.read_output(job_id,name,offset,limit,encoding)


@mcp.tool()
def read_log(job_id: str, name: str='log.lammps', max_rows: StrictInt=2000) -> dict:
    """Parse multiple thermo tables and actual minimizer convergence (<=16 MiB log)."""
    bounded(max_rows,1,10000,'max_rows')
    path=safe_path(jobs.directory(job_id),name,exists=True)
    return parse_thermo(read_bytes(path).decode(errors='replace'),max_rows)


@mcp.tool()
def read_trajectory(job_id: str, name: str='trajectory.lammpstrj', offset_frame: StrictInt=0,
                    max_frames: StrictInt=1, max_atoms: StrictInt=4096) -> dict:
    """Read paginated frames of custom text dumps; retain columns, box flags/bounds and units from script."""
    return science.read_trajectory(safe_path(jobs.directory(job_id),name,exists=True),offset_frame,max_frames,max_atoms)


@mcp.tool()
def restart(job_id: str, restart_name: str='final.restart', action: str='info',
            continuation_script: str='', input_files: list[str] | None=None,
            timeout_seconds: StrictInt=120, mpi_ranks: StrictInt=1) -> dict:
    """Inspect/convert a binary restart or continue with explicit native commands.

    action: info, data, dump, continue. Continuing requires declaring fixes/pair_coeff
    when the chosen LAMMPS style does not store them in restart. Files are copied
    to a fresh job; original files remain unchanged.
    """
    path=safe_path(jobs.directory(job_id),restart_name,exists=True)
    payload=read_bytes(path)
    if action=='info': script='read_restart input.restart\ninfo system\n'
    elif action=='data': script='read_restart input.restart\nwrite_data converted.data nocoeff\n'
    elif action=='dump': script='read_restart input.restart\nwrite_dump all custom converted.lammpstrj id type x y z modify sort id\n'
    elif action=='continue':
        if not continuation_script.strip(): raise ValueError('continuation_script required')
        script='read_restart input.restart\n'+continuation_script
    else: raise ValueError('action must be info, data, dump or continue')
    return jobs.submit(script,input_files,timeout_seconds,mpi_ranks,
                       prepared=lambda directory:(directory/'input.restart').write_bytes(payload))


@mcp.tool()
def single_point(input_name: str, model: str='lj', epsilon: StrictFloat=1., sigma: StrictFloat=1.,
                 cutoff: StrictFloat=3., timeout_seconds: StrictInt=120, mpi_ranks: StrictInt=1) -> dict:
    """Submit energy/forces calculation in metal units; models lj, cu_eam, si_tersoff.

    LJ uses shifted energy, common epsilon/sigma for all types and explicit cutoff.
    Returns job_id, then wait/get_result. Requires periodic atomic input.
    """
    return science.workflow(jobs,input_name,model,'single_point',epsilon=epsilon,sigma=sigma,cutoff=cutoff,
                            timeout_seconds=timeout_seconds,mpi_ranks=mpi_ranks)


@mcp.tool()
def minimize(input_name: str, model: str='lj', steps: StrictInt=1000, fmax: StrictFloat=.01,
             epsilon: StrictFloat=1., sigma: StrictFloat=1., cutoff: StrictFloat=3.,
             timeout_seconds: StrictInt=120, mpi_ranks: StrictInt=1) -> dict:
    """Submit fixed-cell conjugate-gradient minimization; actual stop reason determines convergence."""
    return science.workflow(jobs,input_name,model,'minimize',steps=steps,fmax=fmax,
                            epsilon=epsilon,sigma=sigma,cutoff=cutoff,timeout_seconds=timeout_seconds,mpi_ranks=mpi_ranks)


@mcp.tool()
def run_md(input_name: str, model: str='lj', steps: StrictInt=1000, ensemble: str='nve',
           temperature: StrictFloat=300., timestep: StrictFloat=.001, seed: StrictInt=31415,
           epsilon: StrictFloat=1., sigma: StrictFloat=1., cutoff: StrictFloat=3.,
           timeout_seconds: StrictInt=120, mpi_ranks: StrictInt=1) -> dict:
    """Submit NVE/NVT/NPT MD (metal units, dt in ps). NPT targets 0 bar isotropic pressure.

    Creates fresh seeded Maxwell velocities; use native restart for continuation.
    """
    return science.workflow(jobs,input_name,model,'md',steps=steps,ensemble=ensemble,
                            temperature=temperature,timestep=timestep,seed=seed,epsilon=epsilon,sigma=sigma,
                            cutoff=cutoff,timeout_seconds=timeout_seconds,mpi_ranks=mpi_ranks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport',choices=['stdio','streamable-http'],default='stdio')
    args=parser.parse_args()
    def terminate(signum,frame):
        jobs.close(); raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    try: mcp.run(transport=args.transport)
    finally: jobs.close()


if __name__=='__main__': main()
