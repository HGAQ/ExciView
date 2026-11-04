import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.cube import write_cube, read_cube
import BSE_tools as b
from tools.conditional_workflow import prepare_inputs
from tools.conditional_density import calculate, hole_value, canonical_kpoints, OrbitalCubes
from tools.validate_conditional_producer import validate


class ConditionalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run = Path(self.directory.name)
        self.aims = self.run/'aims'; self.aims.mkdir()
        self.atoms = Atoms('He', positions=[[.2,.3,.4]], cell=[2,3,4], pbc=True)
        self.origin = np.array([-.2,-.3,-.4])
        self.coefficients = np.array([1,1j], complex).reshape(1,2,1,1)/np.sqrt(2)
        self.kpoints = np.array([[0,0,0],[.5,0,0]])
        self.primitive_orbitals = {}
        self.cube_requests = []
        for kpoint in (1,2):
            for band in (1,2):
                orbital_amplitude = np.full((3,4,5), band+.2j*kpoint)
                self.primitive_orbitals[kpoint,band] = orbital_amplitude
                for kind, orbital_part in [('eigenstate',orbital_amplitude.real),('eigenstate_imag',orbital_amplitude.imag),('eigenstate_density',abs(orbital_amplitude)**2)]:
                    filename=f'cube_001_{kind}_{band:05d}_spin_1_k_point_{kpoint:04d}.cube'
                    with (self.aims/filename).open('w') as stream:
                        write_cube(stream,self.atoms,data=orbital_part,origin=self.origin)
                    self.cube_requests.append(dict(kpoint=kpoint,band=band,kind=kind,axis=None,filename=filename))
        (self.run/'cube_requests.json').write_text(json.dumps(self.cube_requests))
        (self.aims/'bz_sampling_out').write_text('2 1 1\n2 2\n1 .5 0 0 0\n2 .5 .5 0 0\n')
        self.bz_hash=hashlib.sha256((self.aims/'bz_sampling_out').read_bytes()).hexdigest()
        self.validation=dict(complex_cubes_valid=True,existing_BSE_gauge_compatible=True,
                             source_hashes={'aims/bz_sampling_out':self.bz_hash})
        (self.run/'orbital_validation.json').write_text(json.dumps(self.validation))
        self.save_archive()

    def tearDown(self):
        self.directory.cleanup()

    def save_archive(self):
        np.savez(self.run/'excitons.npz',coefficients=self.coefficients,kpoints=self.kpoints,
                 kmesh=[2,1,1],v_start=1,c_start=2,source='LibBSE TDA',wavefunction_gauge='native',
                 provenance=json.dumps({str(self.aims/'bz_sampling_out'):self.bz_hash}))

    def arguments(self, output='density'):
        return SimpleNamespace(excitons=self.run/'excitons.npz',orbitals=self.aims,
                               validation=self.run/'orbital_validation.json',state=0,
                               hole=self.origin+[.01,.01,.01],hole_atom=None,output=self.run/output,
                               no_plot=True,save_wavefunction=False)

    def test_menu6_complete_cube_snippet_only(self):
        geometry=self.run/'geometry.in'
        write(geometry,self.atoms,format='aims')
        output=self.run/'generated'
        answers=['6',str(geometry),str(output),'2','1','1','1','1','1','2','3 4 5','-.2 -.3 -.4']
        with patch('builtins.input',side_effect=answers),contextlib.redirect_stdout(io.StringIO()): b.main()
        self.assertEqual({path.name for path in output.iterdir()}, {'cube_cond_snippet.in','cube_requests.json','conditional_setup.json'})
        snippet=(output/'cube_cond_snippet.in').read_text()
        requests=json.loads((output/'cube_requests.json').read_text())
        self.assertEqual(len(requests),24)
        self.assertEqual(snippet.count('output cube '),24)
        self.assertEqual(snippet.count(' cube edge '),72)
        self.assertEqual(len({request['filename'] for request in requests}),24)
        self.assertEqual(len([request for request in requests if request['axis'] is None]),12)
        with self.assertRaisesRegex(ValueError,'already exists'):
            prepare_inputs(geometry,output,[2,1,1],1,1,1,2,[3,4,5],self.origin)

    def test_menu7_independent_bvk_sum_and_normalization(self):
        output=self.run/'menu7'
        answers=['7',str(self.run/'excitons.npz'),'0',str(self.aims),str(self.run/'orbital_validation.json'),str(output),'',*map(str,self.origin+[.01,.01,.01])]
        with patch('builtins.input',side_effect=answers),contextlib.redirect_stdout(io.StringIO()): b.main()
        self.assertTrue((output/'conditional_electron_density.cube').exists())
        with np.load(output/'conditional_density.npz') as archive:
            calculated_density=archive['density']
            density_cell=archive['cell_angstrom']
        expected_amplitudes=[]
        for translation in (-1,0):
            expected_amplitudes.append(sum(self.coefficients[0,k,0,0]*self.primitive_orbitals[k+1,1][0,0,0].conjugate()*self.primitive_orbitals[k+1,2]*np.exp(2j*np.pi*self.kpoints[k,0]*translation) for k in range(2)))
        expected_density=abs(np.concatenate(expected_amplitudes,axis=0))**2
        voxel_volume=abs(np.linalg.det(density_cell))/expected_density.size
        expected_density/=expected_density.sum()*voxel_volume
        np.testing.assert_allclose(calculated_density,expected_density,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(calculated_density.sum()*voxel_volume,1)
        with (output/'conditional_electron_density.cube').open() as stream: cube_record=read_cube(stream)
        np.testing.assert_allclose(cube_record['data'],calculated_density,rtol=5e-6)
        summary=json.loads((output/'summary.json').read_text())
        self.assertLess(summary['parseval_relative_error'],1e-12)

    def test_phase_covariance(self):
        calculate(self.arguments('before'))
        orbital_phases=np.exp(1j*np.array([[.3,.7],[-.4,1.2]]))
        for (kpoint,band),orbital_amplitude in self.primitive_orbitals.items():
            rotated_amplitude=orbital_amplitude*orbital_phases[kpoint-1,band-1]
            for kind,orbital_part in [('eigenstate',rotated_amplitude.real),('eigenstate_imag',rotated_amplitude.imag)]:
                with (self.aims/f'cube_001_{kind}_{band:05d}_spin_1_k_point_{kpoint:04d}.cube').open('w') as stream:
                    write_cube(stream,self.atoms,data=orbital_part,origin=self.origin)
        self.coefficients[0,:,0,0]*=orbital_phases[:,0]/orbital_phases[:,1]
        self.save_archive()
        calculate(self.arguments('after'))
        with np.load(self.run/'before/conditional_density.npz') as before_archive,np.load(self.run/'after/conditional_density.npz') as after_archive:
            np.testing.assert_allclose(before_archive['density'],after_archive['density'],rtol=1e-5,atol=1e-7)

    def test_bloch_interpolation_across_cell_boundary(self):
        orbital_grid=np.ones((4,2,2),complex)
        self.assertAlmostEqual(abs(hole_value(orbital_grid,[.5,0,0],[.875,0,0])),0,places=14)
        np.testing.assert_allclose(hole_value(orbital_grid,[.5,0,0],[1,0,0]),-1,atol=1e-14)

    def test_missing_or_incompatible_cube_and_stale_producer(self):
        (self.aims/self.cube_requests[1]['filename']).unlink()
        with self.assertRaises(FileNotFoundError): calculate(self.arguments())
        (self.aims/'bz_sampling_out').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Producer changed'): calculate(self.arguments())

    def test_manifest_excludes_translation_cubes(self):
        extra_cube=self.aims/'cube_999_eigenstate_00001_spin_1_k_point_0001.cube'
        extra_cube.write_text('not a primitive cube')
        orbital_reader=OrbitalCubes(self.aims)
        np.testing.assert_allclose(orbital_reader.read(0,1),self.primitive_orbitals[1,1])

    def test_generated_manifest_producer_validation(self):
        geometry=self.run/'geometry.in'
        write(geometry,self.atoms,format='aims')
        generated=self.run/'generated'
        prepare_inputs(geometry,generated,[2,1,1],1,1,1,2,[3,4,5],self.origin)
        for filename in ('conditional_setup.json','cube_requests.json'):
            (self.run/filename).write_text((generated/filename).read_text())
        cube_requests=json.loads((self.run/'cube_requests.json').read_text())
        for request in cube_requests:
            orbital_amplitude=self.primitive_orbitals[request['kpoint'],request['band']]
            cube_origin=self.origin.copy()
            if request['axis'] is not None:
                cube_origin+=self.atoms.cell.array[request['axis']]
                orbital_amplitude=orbital_amplitude*np.exp(2j*np.pi*self.kpoints[request['kpoint']-1,request['axis']])
            orbital_part={'eigenstate':orbital_amplitude.real,'eigenstate_imag':orbital_amplitude.imag,'eigenstate_density':abs(orbital_amplitude)**2}[request['kind']]
            with (self.aims/request['filename']).open('w') as stream:
                write_cube(stream,self.atoms,data=orbital_part,origin=cube_origin)
        (self.aims/'band_out').write_text('2\n1\n2\n\n\n1 1\n1 2 -1\n2 0 1\n2 1\n1 2 -1\n2 0 1\n')
        for filename in ('KS_eigenvector_1.dat','basis_out','control.in','geometry.in'):
            (self.aims/filename).write_text('synthetic producer fixture')
        with contextlib.redirect_stdout(io.StringIO()): validate(self.run)
        calculate(self.arguments('validated'))
        orbital_path=self.aims/cube_requests[1]['filename']
        with orbital_path.open('w') as stream:
            write_cube(stream,self.atoms,data=np.ones((3,4,5)),origin=self.origin)
        with self.assertRaisesRegex(ValueError,'Orbital changed'): calculate(self.arguments('stale'))
        with self.assertRaises(ValueError): validate(self.run)

    def test_reject_incomplete_mesh(self):
        with self.assertRaises(ValueError): canonical_kpoints(self.kpoints,np.array([3,1,1]))
        with self.assertRaises(ValueError): canonical_kpoints(np.zeros((2,3)),np.array([2,1,1]))

if __name__=='__main__': unittest.main()
