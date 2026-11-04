import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ase.io import read

from tools.conditional_density import calculate
from tools.libbse_to_exciview import convert


def prepare_inputs(geometry, output, kmesh, nv, nc, v_start, c_start, cube_grid, cube_origin):
    geometry = Path(geometry); output = Path(output)
    for filename in ('cube_cond_snippet.in', 'cube_requests.json', 'conditional_setup.json'):
        if (output/filename).exists(): raise ValueError(f'Output file already exists: {output/filename}')
    primitive_atoms = read(geometry, format='aims')
    primitive_cell = primitive_atoms.cell.array
    if len(primitive_cell) != 3 or primitive_atoms.get_volume() <= 0:
        raise ValueError('A periodic primitive cell is required.')
    for mesh in (kmesh, cube_grid):
        if len(mesh) != 3 or any(int(n) != n or n < 1 for n in mesh):
            raise ValueError('Grid sizes must be three positive integers.')
    if len(cube_origin) != 3 or not np.isfinite(cube_origin).all():
        raise ValueError('Cube origin must contain three finite coordinates.')
    if min(nv, nc, v_start) < 1 or c_start != v_start + nv:
        raise ValueError('Use contiguous occupied and virtual band windows.')
    if 2*(c_start-1) != sum(primitive_atoms.numbers):
        raise ValueError('Band window does not match the neutral closed-shell electron count.')
    nk = int(np.prod(kmesh))
    cube_requests = []
    cube_lines = []
    # Complete band coverage avoids a second SCF with an unrelated band gauge.
    for kpoint in range(1, nk+1):
        for band in range(v_start, c_start+nc):
            for kind in ('eigenstate', 'eigenstate_imag', 'eigenstate_density'):
                cube_requests.append(dict(kpoint=kpoint, band=band, kind=kind, axis=None))
    translation_kpoints = sorted({min(2, nk), (nk+1)//2})
    for kpoint in translation_kpoints:
        for axis in range(3):
            for kind in ('eigenstate', 'eigenstate_imag'):
                cube_requests.append(dict(kpoint=kpoint, band=c_start, kind=kind, axis=axis))
    for cube_index, request in enumerate(cube_requests, 1):
        filename = f"cube_{cube_index:04d}_{request['kind']}_{request['band']:05d}_spin_1_k_point_{request['kpoint']:04d}.cube"
        request['filename'] = filename
        request_origin = np.asarray(cube_origin, dtype=float).copy()
        if request['axis'] is not None: request_origin += primitive_cell[request['axis']]
        cube_lines.append(f"\noutput cube {request['kind']} {request['band']}\n cube kpoint {request['kpoint']}\n cube filename {filename}\n")
        cube_lines.append(' cube origin '+' '.join(f'{coordinate:.12f}' for coordinate in request_origin)+'\n')
        for n, edge in zip(cube_grid, primitive_cell):
            cube_lines.append(f' cube edge {n} '+' '.join(f'{coordinate:.12f}' for coordinate in edge/n)+'\n')
    output.mkdir(parents=True, exist_ok=True)
    (output/'cube_cond_snippet.in').write_text(''.join(cube_lines))
    (output/'cube_requests.json').write_text(json.dumps(cube_requests, indent=2)+'\n')
    setup = dict(kmesh=list(kmesh), grid=list(cube_grid), v_start=v_start, c_start=c_start,
                 nv=nv, nc=nc, atom_numbers=primitive_atoms.numbers.tolist(),
                 translation_kpoints=translation_kpoints, cube_count=len(cube_requests),
                 tda_dimension=nk*nv*nc, source_geometry=str(geometry.resolve()),
                 source_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())
    (output/'conditional_setup.json').write_text(json.dumps(setup, indent=2)+'\n')
    print(f'Success! Generated {len(cube_requests)} cube requests in {output}/cube_cond_snippet.in')
    print('Use the compatible scalar FHI-aims producer with off-Gamma real/imaginary cube support.')
    print('Add the snippet to control.in before the species section; use the requested SCF k grid.')
    print('Generate cubes and the KS coefficients used by BSE in the SAME AIMS invocation.')
    return setup


def generate_inputs():
    geometry = input('Geometry filename: ').strip()
    output = input('Output directory: ').strip()
    nkx = int(input('Kx: ')); nky = int(input('Ky: ')); nkz = int(input('Kz: '))
    nv = int(input('Nv: ')); nc = int(input('Nc: '))
    v_start = int(input('Absolute ID First Valence Band: '))
    c_start = int(input('Absolute ID First Conduction Band: '))
    cube_grid = [int(n) for n in input('Cube grid Nx Ny Nz: ').split()]
    cube_origin = [float(x) for x in input('Cube origin x y z (Ang): ').split()]
    prepare_inputs(geometry, output, [nkx, nky, nkz], nv, nc, v_start, c_start, cube_grid, cube_origin)


def analyze_cond():
    fname = Path(input('BSE directory / LibBSE NPZ filename: ').strip()).resolve()
    state_idx = int(input('State Index: '))
    orbitals = Path(input('Orbital cube directory: ').strip()).resolve()
    validation = Path(input('Orbital validation filename: ').strip()).resolve()
    output = Path(input('Output directory: ').strip()).resolve()
    if output.exists(): raise ValueError('Output directory already exists.')
    if fname.is_dir():
        v_start = int(input('Abs Valence ID: ')); c_start = int(input('Abs Conduction ID: '))
        archive_path = output.parent/(output.name+'_excitons.npz')
        if archive_path.exists(): raise ValueError('Imported archive already exists.')
        output.parent.mkdir(parents=True, exist_ok=True)
        convert(fname, archive_path, v_start, c_start)
        fname = archive_path
    print('\n[Fixed Hole Position]')
    hole_atom_text = input('Hole atom (1-based, blank for coordinates): ').strip()
    hole_atom = int(hole_atom_text) if hole_atom_text else None
    hole_coordinates = None
    if hole_atom is None:
        hx = float(input('  x (Ang): ')); hy = float(input('  y (Ang): ')); hz = float(input('  z (Ang): '))
        hole_coordinates = [hx, hy, hz]
    calculate(SimpleNamespace(excitons=fname, orbitals=orbitals, validation=validation,
                             state=state_idx, hole=hole_coordinates, hole_atom=hole_atom,
                             output=output, no_plot=True, save_wavefunction=False))
