"""Independent numerical checks, executed in a separate image process."""
import json
import math
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase import Atoms

root=Path('/workspace')
atoms=bulk('Cu','fcc',a=3.61,cubic=True).repeat((3,3,3))
saved=read(root/'cu.extxyz')
assert saved.get_chemical_symbols()==atoms.get_chemical_symbols()
assert np.allclose(saved.positions,atoms.positions) and np.allclose(saved.cell,atoms.cell)
atoms.calc=EMT()
energy=float(atoms.get_potential_energy()); forces=atoms.get_forces().tolist()
# Independent pair histogram + spherical-shell volume normalization for total RDF.
distance=neighbor_list('d',atoms,4.0)
hist,edges=np.histogram(distance,bins=80,range=(0,4.0))
shell=4*math.pi/3*np.diff(edges**3)
rdf=hist/(len(atoms)*(len(atoms)/atoms.get_volume())*shell)
# Independent optimization, no import of either MCP server.
dimer=Atoms('Ar2',positions=[[0,0,0],[1.5,0,0]],cell=[10,10,10],pbc=False)
dimer.calc=LennardJones(epsilon=1,sigma=1,rc=3)
initial=float(dimer.get_potential_energy())
optimizer=BFGS(dimer,logfile=None)
converged=bool(optimizer.run(fmax=.01,steps=100))
optimized=read(root/'optimized.extxyz')
assert np.allclose(optimized.positions,dimer.positions,atol=1e-10)
result={'energy_eV':energy,'forces_eV_per_A':forces,'r_A':((edges[:-1]+edges[1:])/2).tolist(),
        'g_r':rdf.tolist(),'coordination':int(len(distance)/len(atoms)),
        'initial_energy_eV':initial,'optimized_energy_eV':float(dimer.get_potential_energy()),
        'optimized_distance_A':float(dimer.get_distance(0,1)), 'converged':converged}
print(json.dumps(result,allow_nan=False))
