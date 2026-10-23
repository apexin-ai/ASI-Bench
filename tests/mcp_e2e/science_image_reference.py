"""Independent references, executed inside a science image without importing MCP."""
import json
import math
import sys
from pathlib import Path

import numpy as np


def gmsh_reference():
    lines = Path('/workspace/box.msh').read_text().splitlines()
    i = lines.index('$Nodes')
    nodes = {int(row[0]): np.array(row[1:], dtype=float) for row in (line.split() for line in lines[i+2:i+2+int(lines[i+1])])}
    i = lines.index('$Elements')
    elements = [list(map(int, line.split())) for line in lines[i+2:i+2+int(lines[i+1])]]
    assert all(row[1] == 4 for row in elements), 'expected tetrahedra'
    volumes = []
    for row in elements:
        a, b, c, d = (nodes[t] for t in row[3+row[2]:])
        volumes.append(abs(float(np.linalg.det(np.stack([b-a, c-a, d-a]))))/6)
    coords = np.array(list(nodes.values()))
    assert np.allclose(coords.min(axis=0), 0) and np.allclose(coords.max(axis=0), 1)
    assert min(volumes) > 0 and math.isclose(sum(volumes), 1, abs_tol=1e-10)
    return {'nodes': len(nodes), 'elements': len(elements), 'volume': sum(volumes), 'min_cell_volume': min(volumes)}


def fenicsx_reference():
    # Assemble PETSc operators explicitly, independently of the adapter's LinearProblem.
    import dolfinx.fem.petsc as fp
    from dolfinx import fem, mesh
    from petsc4py import PETSc
    from mpi4py import MPI
    import ufl
    import subprocess
    import xml.etree.ElementTree as ET
    results = {}
    for n, degree in [(8, 1), (16, 1), (8, 2)]:
        domain = mesh.create_unit_square(MPI.COMM_SELF, n, n)
        V = fem.functionspace(domain, ('Lagrange', degree))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        exact = ufl.sin(math.pi*x[0])*ufl.sin(math.pi*x[1])
        a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
        L = fem.form(2*math.pi**2*exact*v*ufl.dx)
        domain.topology.create_connectivity(1, 2)
        facets = mesh.exterior_facet_indices(domain.topology)
        bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, 1, facets), V)
        A = fp.assemble_matrix(a, bcs=[bc]); A.assemble()
        b = fp.assemble_vector(L)
        fp.apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(b, [bc])
        solver = PETSc.KSP().create(MPI.COMM_SELF)
        solver.setOperators(A); solver.setType('preonly'); solver.getPC().setType('lu')
        uh = fem.Function(V)
        solver.solve(b, uh.x.petsc_vec); uh.x.scatter_forward()
        residual = b.duplicate(); A.mult(uh.x.petsc_vec, residual); residual.axpy(-1, b)
        assert solver.getConvergedReason() > 0 and residual.norm() < 1e-10
        error = math.sqrt(float(fem.assemble_scalar(fem.form((uh-exact)**2*ufl.dx))))
        path = Path(f'/workspace/p{n}d{degree}.xdmf')
        tree = ET.parse(path)
        datasets = []
        arrays = {}
        for item in tree.iter('DataItem'):
            if item.get('Format') == 'HDF':
                filename, dataset = item.text.strip().split(':')
                dumped = subprocess.run(['h5dump', '-d', dataset, '-y', '-w', '0', '-m', '%.17g', str(path.parent/filename)], capture_output=True, text=True, check=True).stdout
                data = np.fromstring(dumped.split('DATA {', 1)[1].split('}', 1)[0], sep=',')
                shape = list(map(int, item.get('Dimensions').split()))
                assert data.size == math.prod(shape) and np.isfinite(data).all()
                arrays[dataset] = data.reshape(shape)
                datasets.append({'path': dataset, 'shape': shape})
        assert len(datasets) >= 3, datasets
        coordinates = next(value for key, value in arrays.items() if key.endswith('/geometry'))
        solution = next(value for key, value in arrays.items() if key.startswith('/Function/')).reshape(-1)
        expected = np.sin(math.pi*coordinates[:, 0])*np.sin(math.pi*coordinates[:, 1])
        assert np.max(np.abs(solution-expected)) < .03
        boundary = (np.isclose(coordinates[:, :2], 0) | np.isclose(coordinates[:, :2], 1)).any(axis=1)
        assert np.max(np.abs(solution[boundary])) < 1e-12
        results[f'{n},{degree}'] = {'l2_error': error, 'residual_norm': residual.norm(), 'datasets': datasets}
        residual.destroy(); solver.destroy(); b.destroy(); A.destroy()
    return results


if __name__ == '__main__':
    print(json.dumps(gmsh_reference() if sys.argv[1] == 'gmsh' else fenicsx_reference()))
