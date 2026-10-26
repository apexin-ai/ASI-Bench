"""Headless OVITO analysis; no Python script modifiers or remote imports."""
from __future__ import annotations
import json
import os
import threading
from typing import Any
from mcp.server.fastmcp import FastMCP
from pydantic import StrictInt, StrictFloat
from common import bounded_int, positive, read_atoms, input_path, output_path, workspace

os.environ.setdefault('QT_QPA_PLATFORM', 'ovitoheadless')
os.environ.setdefault('OVITO_GUI_MODE', '0')

mcp = FastMCP('ASI-Bench OVITO')
_LOCK = threading.RLock()


def _pipeline(input_name):
    # Parse a bounded first frame before OVITO loads it. ASE also validates cell/coordinates.
    read_atoms(input_name)
    from ovito.io import import_file
    path = input_path(input_name)
    formats = {'.xyz': 'xyz', '.extxyz': 'xyz', '.cif': 'cif'}
    return import_file(str(path), input_format=formats[path.suffix.lower()], multiple_frames=False)


def _save(result, name):
    path = output_path(name, ('.json',))
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return {**result, 'analysis_path': str(path)}


@mcp.tool()
def get_info() -> dict[str, Any]:
    """Report the real headless OVITO runtime and analysis scope."""
    import ovito
    return {'ovito_version': ovito.version_string, 'workspace': str(workspace()),
            'analyses': ['polyhedral_template_matching', 'radial_distribution', 'dislocation_analysis'],
            'units': 'length units follow input coordinates (ASE adapter writes angstrom)'}


@mcp.tool()
def inspect_structure(input_name: str) -> dict[str, Any]:
    """Inspect the first frame's particle count, types and simulation cell."""
    with _LOCK:
        data = _pipeline(input_name).compute(0)
        return {'particle_count': data.particles.count,
                'particle_types': [{'id': t.id, 'name': t.name} for t in data.particles.particle_types.types],
                'cell_matrix': data.cell[...].tolist(), 'pbc': list(data.cell.pbc)}


@mcp.tool()
def identify_structure(input_name: str, rmsd_cutoff: StrictFloat = .1,
                       output_name: str = 'structure-types.json') -> dict[str, Any]:
    """Polyhedral template matching: counts/types, RMSD per particle; no GUI required."""
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    threshold = positive(rmsd_cutoff, .001, 1, 'rmsd_cutoff')
    with _LOCK:
        pipeline = _pipeline(input_name)
        modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=threshold, output_rmsd=True)
        pipeline.modifiers.append(modifier)
        data = pipeline.compute(0)
        import numpy as np
        ids = np.asarray(data.particles['Structure Type']).astype(int)
        rmsd = np.asarray(data.particles['RMSD'])
        return _save({'particle_count': len(ids), 'structure_types': ids.tolist(),
                      'counts': {t.name: int((ids == t.id).sum()) for t in modifier.structures},
                      'rmsd': rmsd.tolist()}, output_name)


@mcp.tool()
def compute_rdf(input_name: str, cutoff: StrictFloat = 5, bins: StrictInt = 100,
                output_name: str = 'rdf.json') -> dict[str, Any]:
    """Compute total RDF and coordination numbers for the first frame."""
    from ovito.modifiers import CoordinationAnalysisModifier
    radius = positive(cutoff, .1, 20, 'cutoff')
    count = bounded_int(bins, 2, 512, 'bins')
    with _LOCK:
        pipeline = _pipeline(input_name)
        original = pipeline.compute(0)
        # Keep periodic image multiplicity bounded for dense/tiny cells.
        if not all(original.cell.pbc):
            raise ValueError('RDF requires a fully periodic 3D cell')
        import numpy as np
        cell = np.asarray(original.cell)[:, :3]
        heights = abs(np.linalg.det(cell))/np.linalg.norm(np.cross(cell[:, [1,2,0]].T, cell[:, [2,0,1]].T), axis=1)
        if radius > float(heights.min()) / 2:
            raise ValueError('cutoff must not exceed half the shortest periodic cell height')
        pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=radius, number_of_bins=count))
        data = pipeline.compute(0)
        table = data.tables['coordination-rdf'].xy()
        return _save({'particle_count': data.particles.count, 'cutoff': radius, 'bins': count,
                      'r_A': table[:, 0].tolist(), 'g_r': table[:, 1].tolist(),
                      'coordination': np.asarray(data.particles['Coordination']).astype(int).tolist()}, output_name)


@mcp.tool()
def dislocation_analysis(input_name: str, lattice: str = 'fcc',
                         output_name: str = 'dislocations.json') -> dict[str, Any]:
    """DXA dislocation summary for fcc/bcc/diamond; lengths use input units."""
    from ovito.modifiers import DislocationAnalysisModifier
    choices = {'fcc': DislocationAnalysisModifier.Lattice.FCC,
               'bcc': DislocationAnalysisModifier.Lattice.BCC,
               'diamond': DislocationAnalysisModifier.Lattice.CubicDiamond}
    if lattice not in choices:
        raise ValueError('lattice must be fcc, bcc or diamond')
    with _LOCK:
        pipeline = _pipeline(input_name)
        pipeline.modifiers.append(DislocationAnalysisModifier(input_crystal_structure=choices[lattice]))
        data = pipeline.compute(0)
        segments = data.dislocations.segments
        return _save({'particle_count': data.particles.count, 'lattice': lattice,
                      'segment_count': len(segments),
                      'total_line_length': float(sum(s.length for s in segments)),
                      'segments': [{'length': float(s.length), 'burgers_vector': s.true_burgers_vector.tolist()} for s in segments]}, output_name)


@mcp.tool()
def render_snapshot(input_name: str, width: StrictInt = 640, height: StrictInt = 480,
                    output_name: str = 'snapshot.png') -> dict[str, Any]:
    """CPU Tachyon orthographic snapshot of the first frame; no display/GPU needed."""
    import hashlib
    from ovito.vis import Viewport, TachyonRenderer
    size = (bounded_int(width, 64, 1024, 'width'), bounded_int(height, 64, 1024, 'height'))
    path = output_path(output_name, ('.png',))
    with _LOCK:
        pipeline = _pipeline(input_name)
        pipeline.add_to_scene()
        try:
            view = Viewport(type=Viewport.Type.Ortho, camera_dir=(1, 1, -1))
            view.zoom_all(size=size)
            view.render_image(filename=str(path), size=size, background=(1, 1, 1),
                              renderer=TachyonRenderer())
        finally:
            pipeline.remove_from_scene()
    return {'image_path': str(path), 'width': size[0], 'height': size[1],
            'renderer': 'Tachyon', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    mcp.run(transport='stdio')
