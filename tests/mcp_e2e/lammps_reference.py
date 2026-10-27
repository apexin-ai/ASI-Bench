"""Independent ASE LJ reference and native library EAM/Tersoff calculations; no adapter imports."""
import json
from pathlib import Path
from io import StringIO
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import write
from ase.optimize import BFGS
import lammps

atoms=Atoms('Ar2',positions=[[5,5,5],[6.5,5,5]],cell=[20]*3,pbc=True)
atoms.calc=LennardJones(epsilon=1.,sigma=1.,rc=3.)
answer={'dimer_energy':float(atoms.get_potential_energy()),'dimer_forces':atoms.get_forces().tolist()}
BFGS(atoms,logfile=None).run(fmax=1e-5,steps=200)
answer['optimized_energy']=float(atoms.get_potential_energy())
pot=Path(lammps.__file__).parent/'share/lammps/potentials'
for symbol,lattice,a,file,model,style,coeff in [('Cu','fcc',3.61,'copper.extxyz','cu_eam','eam',str(pot/'Cu_u3.eam')),
                                               ('Si','diamond',5.43,'silicon.extxyz','si_tersoff','tersoff',str(pot/'Si.tersoff')+' Si')]:
    atoms=bulk(symbol,lattice,a=a,cubic=True).repeat((2,2,2))
    buf=StringIO();write(buf,atoms,format='extxyz');answer[file]=buf.getvalue()
    # Construct engine atoms from the independent ASE cell via library commands, not adapter data writer.
    engine=lammps.lammps(cmdargs=['-screen','none','-log','none'])
    engine.commands_string(f'units metal\natom_style atomic\nboundary p p p\nregion box block 0 {2*a} 0 {2*a} 0 {2*a}\ncreate_box 1 box\nmass 1 {atoms.get_masses()[0]}')
    for x,y,z in atoms.positions:engine.command(f'create_atoms 1 single {x:.16g} {y:.16g} {z:.16g} units box')
    engine.commands_string(f'pair_style {style}\npair_coeff * * {coeff}\nrun 0')
    answer[model]=engine.get_thermo('pe');engine.close()
print(json.dumps(answer))
