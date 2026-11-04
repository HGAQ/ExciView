"""Check occupations, complete complex cubes and Bloch translation phases."""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
from ase.io.cube import read_cube

RUN = None
AIMS = None
SETUP = None
ORBITAL_HASHES = {}


def read_orbital_cube(filename):
    ORBITAL_HASHES[filename] = hashlib.sha256((AIMS/filename).read_bytes()).hexdigest()
    with (AIMS / filename).open() as cube_stream:
        orbital_cube = read_cube(cube_stream)
    if orbital_cube['data'].shape != tuple(SETUP['grid']) or not np.isfinite(orbital_cube['data']).all():
        raise ValueError(f'Invalid cube: {filename}')
    return orbital_cube


def validate(run):
    global RUN, AIMS, SETUP
    RUN = Path(run).resolve()
    AIMS = RUN/'aims'
    SETUP = json.loads((RUN/'conditional_setup.json').read_text())
    ORBITAL_HASHES.clear()
    nk_expected = int(np.prod(SETUP['kmesh']))
    v_start, c_start, nc = SETUP['v_start'], SETUP['c_start'], SETUP['nc']
    bz_lines = (AIMS / 'bz_sampling_out').read_text().splitlines()
    assert list(map(int, bz_lines[0].split())) == SETUP['kmesh']
    assert int(bz_lines[1].split()[0]) == nk_expected
    bz_records = np.atleast_2d(np.loadtxt(AIMS / 'bz_sampling_out', skiprows=2))
    assert bz_records.shape[0] == nk_expected
    np.testing.assert_array_equal(bz_records[:, 0], np.arange(1, nk_expected+1))
    np.testing.assert_allclose(bz_records[:, 1], 1/nk_expected, atol=1e-10, rtol=0)
    assert len(np.unique(np.rint(bz_records[:, 2:5]*SETUP['kmesh']).astype(int) % SETUP['kmesh'], axis=0)) == nk_expected
    # Check the actual occupations before assuming the requested occupied/virtual window.
    band_lines = (AIMS / 'band_out').read_text().splitlines()
    nk, nspin, nbands = [int(line) for line in band_lines[:3]]
    assert (nk, nspin) == (nk_expected, 1) and nbands >= c_start+nc-1
    band_cursor = 5
    for kpoint in range(1, nk_expected+1):
        assert list(map(int, band_lines[band_cursor].split())) == [kpoint, 1]
        band_cursor += 1
        band_records = np.array([[float(x) for x in line.split()]
                                for line in band_lines[band_cursor:band_cursor+nbands]])
        np.testing.assert_array_equal(band_records[:, 0], np.arange(1, nbands+1))
        np.testing.assert_allclose(band_records[:c_start-1, 1], 2, atol=1e-5, rtol=0)
        np.testing.assert_allclose(band_records[c_start-1:, 1], 0, atol=1e-5, rtol=0)
        band_cursor += nbands
    cube_requests = json.loads((RUN / 'cube_requests.json').read_text())
    cube_paths = {(r['kpoint'], r['band'], r['kind'], r['axis']): r['filename'] for r in cube_requests}
    max_density_error = 0.
    max_phase_error = 0.
    max_translation_coordinate_error = 0.
    primitive_reference = None
    for kpoint in range(1, nk_expected+1):
        for band in range(v_start, c_start+nc):
            orbital_parts = []
            for kind in ('eigenstate', 'eigenstate_imag', 'eigenstate_density'):
                orbital_cube = read_orbital_cube(cube_paths[kpoint, band, kind, None])
                if primitive_reference is None:
                    primitive_reference = orbital_cube
                for cube_geometry, reference_geometry in (
                    (orbital_cube['origin'], primitive_reference['origin']),
                    (orbital_cube['atoms'].cell.array, primitive_reference['atoms'].cell.array),
                    (orbital_cube['atoms'].positions, primitive_reference['atoms'].positions)):
                    np.testing.assert_allclose(cube_geometry, reference_geometry, atol=2e-6, rtol=0)
                np.testing.assert_array_equal(orbital_cube['atoms'].numbers, SETUP['atom_numbers'])
                orbital_parts.append(orbital_cube['data'])
            orbital_amplitude = orbital_parts[0] + 1j*orbital_parts[1]
            orbital_density = orbital_parts[2]
            max_density_error = max(max_density_error, float(np.max(abs(abs(orbital_amplitude)**2-orbital_density))
                                                             / max(np.max(abs(orbital_density)), 1e-30)))
            if band == c_start and kpoint in SETUP['translation_kpoints']:
                for axis in range(3):
                    shifted_parts = []
                    for kind in ('eigenstate', 'eigenstate_imag'):
                        shifted_cube = read_orbital_cube(cube_paths[kpoint, band, kind, axis])
                        # Cube headers round origins and voxel vectors to six
                        # decimals in bohr; reconstructing a cell multiplies
                        # voxel rounding by the number of grid points.
                        translation_tolerance = (orbital_density.shape[axis] + 2) * 0.5e-6 * 0.5291772105638411
                        translation_coordinate_error = float(np.max(abs(
                            shifted_cube['origin']-primitive_reference['origin']
                            -primitive_reference['atoms'].cell.array[axis])))
                        max_translation_coordinate_error = max(max_translation_coordinate_error, translation_coordinate_error)
                        np.testing.assert_allclose(shifted_cube['origin']-primitive_reference['origin'],
                                                   primitive_reference['atoms'].cell.array[axis], atol=translation_tolerance, rtol=0)
                        shifted_parts.append(shifted_cube['data'])
                    shifted_amplitude = shifted_parts[0] + 1j*shifted_parts[1]
                    expected_amplitude = orbital_amplitude*np.exp(2j*np.pi*bz_records[kpoint-1, 2+axis])
                    max_phase_error = max(max_phase_error, float(np.max(abs(shifted_amplitude-expected_amplitude))
                                                                 / max(np.max(abs(expected_amplitude)), 1e-30)))
    if max_density_error > 3e-5 or max_phase_error > 3e-5:
        raise ValueError(f'Cube validation failed: density={max_density_error}, phase={max_phase_error}')
    # BSE will read these very KS vectors; no independent SCF or phase fitting is used.
    ks_sources = list(AIMS.glob('KS_eigenvector_*.dat'))
    assert ks_sources
    producer_sources = ks_sources + [AIMS/name for name in ('basis_out', 'bz_sampling_out', 'band_out', 'control.in', 'geometry.in')]
    source_hashes = {str(path.relative_to(RUN)): hashlib.sha256(path.read_bytes()).hexdigest() for path in producer_sources}
    validation_report = dict(complex_cubes_valid=True, existing_BSE_gauge_compatible=True,
                             gauge_evidence='Same DFT invocation exports cubes and the KS vectors consumed by BSE.',
                             density_squared_max_relative_error=max_density_error,
                             bloch_translation_max_relative_error=max_phase_error,
                             translation_coordinate_max_error_angstrom=max_translation_coordinate_error,
                             translation_coordinate_tolerance='(grid_axis + 2) * 0.5e-6 bohr, from six-decimal cube headers',
                             source_hashes=source_hashes, orbital_sha256=ORBITAL_HASHES, kmesh=SETUP['kmesh'], grid=SETUP['grid'],
                             occupied_bands=c_start-1, native_orbitals=nk_expected*(SETUP['nv']+nc), cube_count=len(cube_requests))
    (RUN / 'orbital_validation.json').write_text(json.dumps(validation_report, indent=2)+'\n')
    print(json.dumps(validation_report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    validate(parser.parse_args().run)
