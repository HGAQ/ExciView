#!/usr/bin/env python3
"""Coherent, normalized fixed-hole density for native-gauge scalar TDA orbitals.

Consumes existing ExciView NPZ and paired FHI-aims real/imaginary orbital cubes.
No DFT/BSE rerun, gauge fitting, band truncation or primitive-density tiling.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def canonical_kpoints(exciton_kpoints, exciton_mesh):
    """Require a complete Gamma-centered mesh; never invent missing k points."""
    if (exciton_mesh.shape != (3,) or np.any(exciton_mesh < 1)
            or not np.all(exciton_mesh == np.rint(exciton_mesh))):
        raise ValueError('kmesh must contain three positive integers')
    if exciton_kpoints.shape != (int(np.prod(exciton_mesh)), 3) or not np.isfinite(exciton_kpoints).all():
        raise ValueError('Incomplete or non-finite k mesh')
    exciton_exact_kpoints = np.rint(exciton_kpoints * exciton_mesh) / exciton_mesh
    if not np.allclose(exciton_exact_kpoints, exciton_kpoints, atol=1e-10, rtol=0):
        raise ValueError('Only Gamma-centered uniform meshes are supported')
    # Reciprocal-lattice-equivalent k points must count as the same mesh point.
    exciton_indices = np.rint(exciton_exact_kpoints * exciton_mesh).astype(int) % exciton_mesh
    if len(np.unique(exciton_indices, axis=0)) != len(exciton_indices):
        raise ValueError('Duplicate/missing k points')
    return exciton_exact_kpoints


def hole_value(orbital_grid, orbital_kpoint, hole_fractional):
    """Interpolate complex psi, restoring Bloch phase at every wrapped corner.

    Interpolating |psi|^2 would discard the phase needed for interference.
    Works for a hole outside the primitive cube as well as inside it.
    """
    orbital_shape = np.asarray(orbital_grid.shape)
    hole_index_float = hole_fractional * orbital_shape
    hole_lower = np.floor(hole_index_float).astype(int)
    hole_fraction = hole_index_float - hole_lower
    hole_amplitude = 0j
    for hole_offset in itertools.product((0, 1), repeat=3):
        hole_offset = np.asarray(hole_offset)
        hole_index = hole_lower + hole_offset
        hole_translation = np.floor_divide(hole_index, orbital_shape)
        hole_weight = np.prod(np.where(hole_offset, hole_fraction, 1 - hole_fraction))
        hole_amplitude += (hole_weight * orbital_grid[tuple(hole_index % orbital_shape)]
                           * np.exp(2j * np.pi * np.dot(orbital_kpoint, hole_translation)))
    return hole_amplitude


def extend_amplitude(exciton_partial, exciton_kpoints, exciton_mesh):
    """Sum Bloch-translated amplitudes, not already-squared primitive densities."""
    orbital_shape = np.asarray(exciton_partial.shape[1:])
    # A complete centered BvK supercell retains the interference between k points.
    bvk_start = -(exciton_mesh // 2)
    bvk_amplitude = np.empty(tuple(orbital_shape * exciton_mesh), dtype=complex)
    for bvk_index in itertools.product(*(range(int(n)) for n in exciton_mesh)):
        bvk_translation = np.asarray(bvk_index) + bvk_start
        bvk_phases = np.exp(2j * np.pi * (exciton_kpoints @ bvk_translation))
        bvk_slices = tuple(slice(i*n, (i+1)*n) for i, n in zip(bvk_index, orbital_shape))
        bvk_amplitude[bvk_slices] = np.einsum('k,kxyz->xyz', bvk_phases, exciton_partial)
    return bvk_amplitude, bvk_start


def cube_origin(orbital_path):
    """Read the standard FHI-aims cube origin in Angstrom without its grid."""
    from ase.units import Bohr
    with orbital_path.open() as orbital_stream:
        next(orbital_stream)
        next(orbital_stream)
        return np.asarray(next(orbital_stream).split()[1:4], dtype=float) * Bohr


class OrbitalCubes:
    """Check grid/geometry compatibility while reading one orbital at a time."""
    def __init__(self, directory):
        self.directory = directory
        self.reference = None
        self.hashes = {}

    def paths(self, k_index, band, orbital_kind):
        orbital_paths = sorted(self.directory.glob(
            f'cube_*_{orbital_kind}_{band:05d}_spin_1_k_point_{k_index+1:04d}.cube'))
        if not orbital_paths:
            raise FileNotFoundError(f'Missing cube for k={k_index+1}, band={band}, {orbital_kind}')
        if self.reference is None:
            # Generated request numbers place primitive cubes before translated copies.
            return orbital_paths[:1]
        # Translated validation cubes must not replace the primitive orbital.
        primitive_paths = []
        for orbital_path in orbital_paths:
            if np.allclose(cube_origin(orbital_path), self.reference['origin'], atol=1e-6, rtol=0):
                primitive_paths.append(orbital_path)
        return primitive_paths

    def read(self, k_index, band):
        from ase.io.cube import read_cube
        orbital_parts = []
        for orbital_kind in ('eigenstate', 'eigenstate_imag'):
            orbital_paths = self.paths(k_index, band, orbital_kind)
            if len(orbital_paths) != 1:
                raise ValueError(f'Expected exactly one cube for k={k_index+1}, band={band}, {orbital_kind}')
            orbital_path = orbital_paths[0]
            self.hashes[orbital_path.name] = hashlib.sha256(orbital_path.read_bytes()).hexdigest()
            with orbital_path.open() as orbital_stream:
                orbital_cube = read_cube(orbital_stream)
            if self.reference is None:
                self.reference = orbital_cube
                if len(self.paths(k_index, band, orbital_kind)) != 1:
                    raise ValueError("Duplicate primitive orbital cubes")
            orbital_reference = self.reference
            if (orbital_cube['data'].shape != orbital_reference['data'].shape
                    or not np.isfinite(orbital_cube['data']).all()
                    or not np.array_equal(orbital_cube['atoms'].numbers, orbital_reference['atoms'].numbers)):
                raise ValueError(f'Incompatible orbital grid/atoms: {orbital_path.name}')
            for orbital_geometry, reference_geometry in (
                (orbital_cube['origin'], orbital_reference['origin']),
                (orbital_cube['atoms'].cell.array, orbital_reference['atoms'].cell.array),
                (orbital_cube['atoms'].positions, orbital_reference['atoms'].positions)):
                if not np.allclose(orbital_geometry, reference_geometry, atol=1e-6, rtol=0):
                    raise ValueError(f'Incompatible orbital geometry: {orbital_path.name}')
            orbital_parts.append(orbital_cube['data'])
        return orbital_parts[0] + 1j * orbital_parts[1]


def select_hole(primitive_atoms, hole_coordinates=None, hole_atom=None):
    """Atom selection is point conditioning at the nucleus, not a band projection.

    Preserve the requested coordinate exactly. Never silently move a hole off
    an atomic node to obtain a visually stronger conditional distribution.
    """
    if (hole_coordinates is None) == (hole_atom is None):
        raise ValueError('Choose exactly one of hole coordinates or a one-based atom index')
    if hole_atom is not None:
        if hole_atom < 1 or hole_atom > len(primitive_atoms):
            raise ValueError('Hole atom index must be in 1..number of primitive atoms')
        hole_cartesian = primitive_atoms.positions[hole_atom-1].copy()
        hole_selection = dict(mode='atom', atom_index_one_based=int(hole_atom),
                              element=primitive_atoms[hole_atom-1].symbol,
                              offset_angstrom=[0., 0., 0.])
    else:
        hole_cartesian = np.asarray(hole_coordinates, dtype=float)
        hole_selection = dict(mode='coordinate')
    if hole_cartesian.shape != (3,) or not np.isfinite(hole_cartesian).all():
        raise ValueError('Hole must have three finite Cartesian Angstrom coordinates')
    return hole_cartesian, hole_selection


def calculate(args):
    with np.load(args.excitons, allow_pickle=False) as exciton_archive:
        if str(exciton_archive['source']) != 'LibBSE TDA' or str(exciton_archive['wavefunction_gauge']) != 'native':
            raise ValueError('Only scalar native-gauge LibBSE TDA archives are supported')
        if args.state < 0 or args.state >= len(exciton_archive['coefficients']):
            raise ValueError('State index is out of range (zero-based)')
        if 'source_paths' in exciton_archive or 'provenance' in exciton_archive:
            if 'source_paths' in exciton_archive:
                exciton_provenance = dict(zip(exciton_archive['source_paths'].tolist(),
                                             exciton_archive['source_sha256'].tolist()))
            else:
                # Read existing archives; new archives use native NPZ fields.
                exciton_provenance = json.loads(str(exciton_archive['provenance']))
            # Hashes detect changed inputs; they do not independently prove KS/BSE phase agreement.
            for source_name, source_hash in exciton_provenance.items():
                if hashlib.sha256(Path(source_name).read_bytes()).hexdigest() != source_hash:
                    raise ValueError(f'BSE source changed after import: {source_name}')
            bz_path = str((args.orbitals/'bz_sampling_out').resolve())
            if bz_path not in exciton_provenance:
                raise ValueError('Archive and orbital cubes use different producer directories')
        else:
            raise ValueError('LibBSE archive has no source provenance')
        exciton_coefficients = exciton_archive['coefficients'][args.state].copy()
        exciton_mesh = exciton_archive['kmesh'].copy()
        exciton_kpoints = canonical_kpoints(exciton_archive['kpoints'], exciton_mesh)
        valence_start, conduction_start = int(exciton_archive['v_start']), int(exciton_archive['c_start'])
    if (exciton_coefficients.ndim != 3 or exciton_coefficients.shape[0] != len(exciton_kpoints)
            or not np.isfinite(exciton_coefficients).all()
            or not np.isclose(np.sum(abs(exciton_coefficients)**2), 1, atol=2e-6, rtol=0)):
        raise ValueError('Invalid TDA amplitudes or state normalization')
    exciton_mesh = exciton_mesh.astype(int)
    nk, nv, nc = exciton_coefficients.shape
    if min(nv, nc) < 1 or valence_start < 1 or conduction_start != valence_start + nv:
        raise ValueError('Expected contiguous occupied and virtual band windows')
    orbital_reader = OrbitalCubes(args.orbitals)
    first_orbital = orbital_reader.read(0, valence_start)
    primitive_cell = orbital_reader.reference['atoms'].cell.array
    primitive_origin = orbital_reader.reference['origin']
    hole_cartesian, hole_selection = select_hole(
        orbital_reader.reference['atoms'], args.hole, args.hole_atom)
    # Interpolation coordinates are relative to the cube origin, not the crystal origin.
    hole_fractional = (hole_cartesian - primitive_origin) @ np.linalg.inv(primitive_cell)
    orbital_shape = np.asarray(first_orbital.shape)
    exciton_partial = np.zeros((nk, *orbital_shape), dtype=complex)
    hole_orbitals = np.empty((nk, nv), dtype=complex)
    # Read one k point at a time: no need to hold all conduction orbitals in RAM.
    for k_index in range(nk):
        for v_index in range(nv):
            valence_orbital = (first_orbital if k_index == 0 and v_index == 0
                              else orbital_reader.read(k_index, valence_start + v_index))
            hole_orbitals[k_index, v_index] = hole_value(valence_orbital, exciton_kpoints[k_index], hole_fractional)
        # Psi_S(re,rh) = sum_kvc A_S(k,v,c) psi_c,k(re) conjugate(psi_v,k(rh)).
        # Contract valence bands first, preserving all complex interference terms.
        conduction_weights = exciton_coefficients[k_index].T @ hole_orbitals[k_index].conj()
        for c_index in range(nc):
            conduction_orbital = orbital_reader.read(k_index, conduction_start + c_index)
            exciton_partial[k_index] += conduction_weights[c_index] * conduction_orbital
    bvk_amplitude, bvk_start = extend_amplitude(exciton_partial, exciton_kpoints, exciton_mesh)
    voxel_volume = abs(np.linalg.det(primitive_cell)) / np.prod(orbital_shape)
    cond_density = abs(bvk_amplitude)**2
    raw_integral = float(cond_density.sum() * voxel_volume)
    if not np.isfinite(raw_integral) or raw_integral <= 0:
        raise ValueError('Conditional density is undefined at this hole (zero/non-finite norm)')
    # Full BvK translation sum cancels cross-k terms: an independent phase check.
    parseval_integral = float(nk * np.sum(abs(exciton_partial)**2) * voxel_volume)
    if not np.isclose(raw_integral, parseval_integral, rtol=1e-11, atol=0):
        raise ValueError('BvK/Parseval check failed')
    # Normalize over electron coordinates at this fixed hole; keep its raw weight separately.
    cond_density /= raw_integral
    cond_origin = primitive_origin + bvk_start @ primitive_cell
    cond_cell = primitive_cell * exciton_mesh[:, None]
    cond_atoms = orbital_reader.reference['atoms'].repeat(tuple(exciton_mesh))
    cond_atoms.positions += bvk_start @ primitive_cell
    cond_summary = dict(state=args.state, hole_angstrom=hole_cartesian.tolist(),
                        hole_selection=hole_selection,
                        hole_orbital_norm_squared=float(np.sum(abs(hole_orbitals)**2)),
                        kmesh=exciton_mesh.tolist(),
                        primitive_grid=orbital_shape.tolist(), bvk_grid=list(cond_density.shape),
                        valence_bands=list(range(valence_start, valence_start+nv)),
                        conduction_bands=list(range(conduction_start, conduction_start+nc)),
                        density_units='Angstrom^-3', raw_integral=raw_integral,
                        normalized_integral=float(cond_density.sum()*voxel_volume),
                        parseval_relative_error=abs(raw_integral/parseval_integral-1),
                        excitons_sha256=hashlib.sha256(args.excitons.read_bytes()).hexdigest(),
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        orbital_sha256=orbital_reader.hashes,
                        bse_source_hashes=exciton_provenance,
                        gauge_evidence='Source hashes checked; AO-gauge compatibility is not independently verified',
                        limitations='Scalar TDA, complete Gamma-centered mesh, spin-1 cubes; no physical convergence claim')
    cond_optional_arrays = {}
    if getattr(args, 'save_wavefunction', False):
        cond_optional_arrays['amplitude'] = bvk_amplitude / np.sqrt(raw_integral)
    # Native NPZ fields keep metadata readable with allow_pickle=False and no JSON records.
    cond_metadata = {key: value for key, value in cond_summary.items()
                     if key not in ('state', 'hole_angstrom', 'hole_selection',
                                    'orbital_sha256', 'bse_source_hashes')}
    cond_metadata.update(hole_selection_mode=hole_selection['mode'],
                         hole_atom_index=hole_selection.get('atom_index_one_based', 0),
                         orbital_paths=list(orbital_reader.hashes),
                         orbital_sha256=list(orbital_reader.hashes.values()),
                         source_paths=list(exciton_provenance),
                         source_sha256=list(exciton_provenance.values()))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'conditional_density.npz', density=cond_density,
                        origin_angstrom=cond_origin, cell_angstrom=cond_cell,
                        hole_angstrom=hole_cartesian, hole_orbital_values=hole_orbitals,
                        kpoints=exciton_kpoints, atom_positions_angstrom=cond_atoms.positions,
                        atom_numbers=cond_atoms.numbers, state=args.state,
                        **cond_metadata, **cond_optional_arrays)
    print(f'Wrote {args.output}: integral={cond_summary["normalized_integral"]:.15g}, '
          f'Parseval error={cond_summary["parseval_relative_error"]:.3g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--excitons', type=Path, required=True)
    parser.add_argument('--orbitals', type=Path, required=True)
    parser.add_argument('--state', type=int, default=0)
    hole_group = parser.add_mutually_exclusive_group(required=True)
    hole_group.add_argument('--hole', type=float, nargs=3, help='Cartesian Angstrom coordinates')
    hole_group.add_argument('--hole-atom', type=int, help='One-based primitive atom index from orbital cube geometry')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--save-wavefunction', action='store_true', help='Include normalized complex amplitude in the same conditional_density.npz')
    calculate(parser.parse_args())


if __name__ == '__main__':
    main()
