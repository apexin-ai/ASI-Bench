"""Scientific contracts and typed workflow builders; native scripts remain available."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
from runtime import bounded, finite, safe_path, read_bytes, parse_thermo


def potentials_dir():
    import lammps
    return Path(lammps.__file__).parent / 'share/lammps/potentials'


def list_potentials():
    root = potentials_dir()
    return [{'name': p.name, 'size_bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.iterdir()) if p.is_file()]


def capabilities(binary):
    result = subprocess.run([binary, '-h'], capture_output=True, text=True, timeout=20)
    if result.returncode: raise RuntimeError(f'LAMMPS engine startup failed: {result.stderr[-2000:]}')
    text = result.stdout
    match = re.search(r'LAMMPS - (.+)', text)
    section = text.split('Installed packages:', 1)
    packages = section[1].lstrip().split('\n\n', 1)[0].split() if len(section) == 2 else []
    return {'version': match.group(1) if match else text.splitlines()[1],
            'installed_packages': packages,
            'high_level_models': ['lj', 'cu_eam', 'si_tersoff'],
            'high_level_units': {'length': 'angstrom', 'energy': 'eV', 'force': 'eV/angstrom',
                                 'time': 'ps', 'temperature': 'K', 'pressure': 'bar'},
            'native_scripts': True, 'limits': {'concurrent_jobs': 2, 'mpi_ranks': 4,
                'timeout_seconds': 600, 'output_bytes_per_job': 128*1024*1024,
                'high_level_atoms': 4096, 'high_level_md_steps': 100000},
            'trust_boundary': 'Native scripts have the server container permissions, filesystem and network.'}


def workflow(jobs, input_name, model, mode, *, epsilon=1., sigma=1., cutoff=3.,
             steps=100, fmax=.01, temperature=300., timestep=.001, ensemble='nve', seed=31415,
             timeout_seconds=120, mpi_ranks=1):
    from ase.io import read, write
    import numpy as np
    path = safe_path(jobs.workspace, input_name, exists=True)
    if path.suffix not in {'.xyz', '.extxyz', '.cif', '.data'}:
        raise ValueError('input must be XYZ/extxyz/CIF or atomic-style LAMMPS .data')
    read_bytes(path)
    if path.suffix in {'.xyz', '.extxyz'}:
        with path.open() as handle: bounded(int(handle.readline()), 1, 4096, 'atom count')
    atoms = read(str(path), format={'.xyz':'extxyz','.extxyz':'extxyz','.cif':'cif','.data':'lammps-data'}[path.suffix],
                 **({'atom_style':'atomic'} if path.suffix == '.data' else {}))
    bounded(len(atoms), 1, 4096, 'atom count')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('nonfinite structure')
    if not atoms.pbc.all() or atoms.cell.rank != 3 or atoms.cell.volume < 1e-8:
        raise ValueError('high-level tools require a fully periodic nondegenerate 3D cell; use native scripts for other boundaries')
    symbols = sorted(set(atoms.get_chemical_symbols()))
    if model == 'lj':
        finite(epsilon, 1e-8, 100, 'epsilon'); finite(sigma, .01, 100, 'sigma'); finite(cutoff, .1, 100, 'cutoff')
        potential = f'pair_style lj/cut {cutoff}\npair_coeff * * {epsilon} {sigma}\npair_modify shift yes'
    elif model == 'cu_eam':
        if symbols != ['Cu']: raise ValueError('cu_eam supports Cu only')
        potential = f'pair_style eam\npair_coeff * * {potentials_dir() / "Cu_u3.eam"}'
    elif model == 'si_tersoff':
        if symbols != ['Si']: raise ValueError('si_tersoff supports Si only')
        potential = f'pair_style tersoff\npair_coeff * * {potentials_dir() / "Si.tersoff"} Si'
    else: raise ValueError('model must be lj, cu_eam or si_tersoff; use native scripts for other potentials')
    bounded(steps, 1, 100000 if mode == 'md' else 10000, 'steps')
    if mode == 'minimize':
        finite(fmax, 1e-8, 10, 'fmax')
        action = f'min_style cg\nmin_modify norm max\nminimize 0.0 {fmax} {steps} {steps*20}'
    elif mode == 'md':
        finite(temperature, .001, 10000, 'temperature'); finite(timestep, 1e-7, .01, 'timestep')
        bounded(seed, 1, 900000000, 'seed')
        if ensemble not in {'nve', 'nvt', 'npt'}: raise ValueError('ensemble must be nve, nvt or npt')
        if ensemble == 'npt' and not atoms.cell.orthorhombic: raise ValueError('high-level npt requires an orthorhombic cell')
        integrator = {'nve':'nve', 'nvt':f'nvt temp {temperature} {temperature} {100*timestep}',
                      'npt':f'npt temp {temperature} {temperature} {100*timestep} iso 0.0 0.0 {1000*timestep}'}[ensemble]
        action = f'velocity all create {temperature} {seed} mom yes rot no dist gaussian\ntimestep {timestep}\nfix integrate all {integrator}\nrun {steps}'
    elif mode == 'single_point': action = 'run 0'
    else: raise ValueError('unknown workflow')
    script = f'''units metal
atom_style atomic
boundary p p p
read_data structure.data
{potential}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
thermo {max(1, steps//100)}
thermo_style custom step atoms pe ke etotal temp press vol fmax
thermo_modify norm no format float %.16g
# Both coordinates and forces are persisted at the final integration state.
dump trajectory all custom {max(1, steps//100)} trajectory.lammpstrj id type x y z fx fy fz
dump_modify trajectory sort id format float %.16g
{action}
write_dump all custom final.lammpstrj id type x y z fx fy fz modify sort id format float %.16g
write_data final.data
write_restart final.restart
'''
    def prepared(directory):
        write(str(directory/'structure.data'), atoms, format='lammps-data', atom_style='atomic',
              specorder=symbols, masses=True)
        (directory/'workflow.json').write_text(json.dumps({'mode':mode, 'model':model, 'symbols_by_type':symbols,
             'units':'metal', 'atom_count':len(atoms), 'requested_steps':steps,
             'potential_sha256': (hashlib.sha256((potentials_dir()/({'cu_eam':'Cu_u3.eam','si_tersoff':'Si.tersoff'}[model])).read_bytes()).hexdigest()
                                  if model != 'lj' else None)}))
    return jobs.submit(script, timeout_seconds=timeout_seconds, mpi_ranks=mpi_ranks, prepared=prepared)


def read_trajectory(path, offset_frame=0, max_frames=1, max_atoms=4096):
    bounded(offset_frame, 0, 100000, 'offset_frame'); bounded(max_frames, 1, 10, 'max_frames')
    bounded(max_atoms, 1, 4096, 'max_atoms')
    lines = iter(read_bytes(path).decode().splitlines())
    frames=[]; count=0
    while True:
        try: header=next(lines)
        except StopIteration: break
        if not header.strip(): continue
        if header != 'ITEM: TIMESTEP': raise ValueError('expected LAMMPS custom text dump')
        try:
            step=int(next(lines))
            if next(lines) != 'ITEM: NUMBER OF ATOMS': raise ValueError('missing atom count')
            n=int(next(lines)); bounded(n, 1, 10000000, 'dump atom count')
            box_header=next(lines)
            if not box_header.startswith('ITEM: BOX BOUNDS'): raise ValueError('missing box bounds')
            box=[[float(x) for x in next(lines).split()] for _ in range(3)]
            columns=next(lines)
            if not columns.startswith('ITEM: ATOMS '): raise ValueError('missing atom columns')
            columns=columns.split()[2:]
            selected = offset_frame <= count < offset_frame+max_frames
            rows=[]
            for index in range(n):
                values=[float(x) for x in next(lines).split()]
                if len(values)!=len(columns) or not all(__import__('math').isfinite(x) for x in values):
                    raise ValueError('invalid dump row')
                if selected and index < max_atoms: rows.append(values)
            if selected:
                frames.append({'step':step,'atom_count':n,'box_header':box_header,'box_bounds':box,
                    'columns':columns,'rows':rows,'atoms_truncated':n>max_atoms})
            count+=1
            if len(frames) >= max_frames: break
        except StopIteration: raise ValueError('truncated dump') from None
    return {'frames':frames,'next_frame':offset_frame+len(frames)}


def result(jobs, job_id):
    status=jobs.status(job_id); directory=jobs.directory(job_id)
    answer={'status':status, **jobs.files(job_id)}
    log=safe_path(directory,'log.lammps')
    if log.is_file(): answer['thermo']=parse_thermo(read_bytes(log).decode(errors='replace'))
    meta=safe_path(directory,'workflow.json')
    if meta.is_file(): answer['workflow']=json.loads(read_bytes(meta))
    final=safe_path(directory,'final.lammpstrj')
    if final.is_file() and status['state']=='completed':
        answer['final_structure_and_forces']=read_trajectory(final)
    # Native scripts choose their own units/normalization; never invent eV fields for them.
    return answer
