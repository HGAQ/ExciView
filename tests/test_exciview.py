"""Executable analytic regressions for every ExciView menu pipeline."""
import contextlib
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io.cube import read_cube, write_cube
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import BSE_tools as b
from tools.libbse_to_exciview import read_amplitudes, read_mulliken, convert


class ExciViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.cwd=Path.cwd();os.chdir(self.tmp.name)
        self.atoms=Atoms('Si2',positions=[[0,0,0],[.5,.5,.5]],cell=[[2,0,0],[.2,2,0],[.1,.2,2]],pbc=True)
        self.origin=np.array([-.2,-.3,-.4])
        self.coeff=np.array([1,1j,-.5,.5j],complex).reshape(1,2,1,2)/np.sqrt(2.5)
        np.savez('states.npz',coefficients=self.coeff,kpoints=[[0,0,0],[.5,0,0]],kmesh=[2,1,1],v_start=1,c_start=2,wavefunction_gauge='native')
        grid=np.indices((3,4,5)).sum(axis=0)*.03
        self.orbs={}
        for k in (1,2):
            for band in (1,2,3):
                psi=(band*.1+grid)+1j*(k*.2-grid*.3)
                self.orbs[k,band]=psi
                for kind,data in [('eigenstate',psi.real),('eigenstate_imag',psi.imag),('eigenstate_density',abs(psi)**2)]:
                    with open(f'cube_001_{kind}_{band:05d}_spin_1_k_point_{k:04d}.cube','w') as f:
                        write_cube(f,self.atoms,data=data,origin=self.origin)
        self.common=['states.npz','0','1','2','2']

    def tearDown(self):
        os.chdir(self.cwd);self.tmp.cleanup()

    def menu(self,choice,answers):
        with patch('builtins.input',side_effect=[str(choice),*map(str,answers)]),contextlib.redirect_stdout(io.StringIO()) as out:
            b.main()
        return out.getvalue()

    def cube(self,name):
        with open(name) as f:return read_cube(f)

    def mulliken_block(self, kx, shift):
        text = f'k point number: 1: ( {kx} 0 0 )\n'
        for band in (1, 2, 3):
            pop = .1*band + shift
            text += (f'State {band}\n'
                     f'{band} 0 1 1 {pop} {pop*.4} {pop*.6}\n'
                     f'{band} 0 1 2 {1-pop} {(1-pop)*.4} {(1-pop)*.6}\n')
        return text

    def test_multik_mulliken_coordinates_duplicates_and_menu(self):
        Path('mlk').mkdir()
        # Reverse file order and use a periodic image of the second k point.
        Path('mlk/bandmlk1001.out').write_text(
            self.mulliken_block(-.5, .2) + self.mulliken_block(0, 0))
        Path('mlk/bandmlk1002.out').write_text(self.mulliken_block(.5, .2))
        data, files = read_mulliken('mlk', [[0,0,0],[.5,0,0]], [1,2,3])
        self.assertEqual(len(files), 2)
        np.testing.assert_allclose(data['mulliken_populations'][:,:,0,0],
                                   [[.1,.2,.3],[.3,.4,.5]])
        with np.load('states.npz') as archive:
            saved = dict(archive)
        np.savez('states.npz', **saved, **data)
        log = self.menu(2, self.common)
        self.assertIn('Using embedded Mulliken data', log)
        report = Path('exciton_analysis_state_0.dat').read_text()
        # Independent weighted sums: k weights .8/.2, equal c weights within k.
        self.assertIn('Hole Pop: 0.140000', report)
        self.assertIn('Elec Pop: 0.290000', report)
        self.assertIn('H: 0.056000   E: 0.116000', report)

    def test_multik_mulliken_rejects_incomplete_conflicting_or_wrong_grid(self):
        Path('mlk').mkdir()
        path = Path('mlk/bandmlk1001.out')
        path.write_text(self.mulliken_block(0, 0))
        with self.assertRaisesRegex(ValueError, 'Incomplete Mulliken k grid'):
            read_mulliken('mlk', [[0,0,0],[.5,0,0]], [1,2,3])
        path.write_text(self.mulliken_block(0, 0)+self.mulliken_block(0, .2))
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            read_mulliken('mlk', [[0,0,0]], [1,2,3])
        path.write_text(self.mulliken_block(.25, 0))
        with self.assertRaisesRegex(ValueError, 'does not uniquely match'):
            read_mulliken('mlk', [[0,0,0],[.5,0,0]], [1,2,3])
        path.write_text(self.mulliken_block(0, 0))
        with self.assertRaisesRegex(ValueError, 'Missing requested Mulliken bands'):
            read_mulliken('mlk', [[0,0,0]], [1,2,3,4])
        path.write_text(self.mulliken_block(0, 0).replace('0.1 ', 'nan ', 1))
        with self.assertRaisesRegex(ValueError, 'Invalid/duplicate'):
            read_mulliken('mlk', [[0,0,0]], [1,2,3])

    def test_all_seven_menus_and_independent_sums(self):
        self.menu(1,self.common+[2,1,1,0])
        self.assertEqual(b.parse_snippet_for_mapping('mulliken_snippet.in'),[0,1])
        for k in (1,2):
            text=''
            for band in (1,2,3):
                text+=f'State {band}\n{band} 0 1 1 0.25 0.1 0.15\n{band} 0 1 2 0.75 0.4 0.35\n'
            Path(f'bandmlk{1000+k}.out').write_text(text)
        self.menu(2,self.common+['',1,2,'bandmlk{}.out',1001])
        text=Path('exciton_analysis_state_0.dat').read_text()
        self.assertIn('Hole Pop: 0.250000',text)
        self.assertIn('Elec Pop: 0.750000',text)
        self.menu(3,self.common+[1,2,0])
        self.assertEqual(len(b.parse_control_for_cubes()),6)
        self.menu(4,self.common+[1,2,'',''])
        w=abs(self.coeff[0])**2
        for name,marg,start in [('hole',w.sum(2),1),('elec',w.sum(1),2)]:
            expected=sum(marg[k,j]*abs(self.orbs[k+1,start+j])**2 for k,j in np.ndindex(marg.shape))/marg.sum()
            content=self.cube(f'avg_{name}_state_0.cube')
            np.testing.assert_allclose(content['data'],expected,rtol=4e-6,atol=1e-7)
            np.testing.assert_allclose(content['origin'],self.origin,atol=1e-6)
        self.assertIn('Total Norm Captured: 1.000000',self.menu(5,self.common+[2,1,1]))
        self.menu(6,self.common+[1,2,0])
        self.assertEqual(Path('cube_cond_snippet.in').read_text().count('output cube'),12)
        hole_coord=self.origin+np.array([.01,.01,.01])
        self.menu(7,self.common+[1,2,*hole_coord,'',0])
        psi=sum(self.coeff[0,k,0,c]*self.orbs[k+1,1][0,0,0].conjugate()*self.orbs[k+1,c+2] for k in range(2) for c in range(2))
        np.testing.assert_allclose(self.cube('cond_density_state_0.cube')['data'],abs(psi)**2,rtol=5e-6,atol=1e-7)

    def test_threshold_selection_and_retained_normalization(self):
        self.menu(1,self.common+[2,1,1,30])
        self.assertEqual(b.parse_snippet_for_mapping('mulliken_snippet.in'),[0])
        self.menu(3,self.common+[1,2,25])
        self.assertEqual(b.parse_control_for_cubes(),{(1,1),(1,2),(1,3)})
        text=self.menu(4,self.common+[1,2,'',''])
        self.assertIn('captured BSE weight=0.8000000000',text)
        expected=(abs(self.orbs[1,2])**2+abs(self.orbs[1,3])**2)/2
        np.testing.assert_allclose(self.cube('avg_elec_state_0.cube')['data'],expected,rtol=4e-6,atol=1e-7)
        self.menu(6,self.common+[1,2,25])
        self.assertEqual(Path('cube_cond_snippet.in').read_text().count('output cube'),6)
        self.menu(7,self.common+[1,2,*(self.origin+[.01,.01,.01]),'',25])
        psi=sum(self.coeff[0,0,0,c]*self.orbs[1,1][0,0,0].conjugate()*self.orbs[1,c+2] for c in range(2))
        np.testing.assert_allclose(self.cube('cond_density_state_0.cube')['data'],abs(psi)**2,rtol=5e-6,atol=1e-7)

    def test_complex_phase_covariance(self):
        coord=self.origin+np.array([.01,.01,.01])
        self.menu(7,self.common+[1,2,*coord,'',0])
        ref=self.cube('cond_density_state_0.cube')['data']
        phases=np.exp(1j*np.array([[.3,.7,-.4],[1.2,-.8,.5]]))
        for (k,band),orb in self.orbs.items():
            rotated=orb*phases[k-1,band-1]
            for kind,data in [('eigenstate',rotated.real),('eigenstate_imag',rotated.imag)]:
                with open(f'cube_001_{kind}_{band:05d}_spin_1_k_point_{k:04d}.cube','w') as f:write_cube(f,self.atoms,data=data,origin=self.origin)
        rotated=self.coeff.copy()
        for k in range(2):
            for c in range(2):rotated[0,k,0,c]*=phases[k,0]/phases[k,c+1]
        np.savez('states.npz',coefficients=rotated,kpoints=[[0,0,0],[.5,0,0]],kmesh=[2,1,1],v_start=1,c_start=2,wavefunction_gauge='native')
        self.menu(7,self.common+[1,2,*coord,'',0])
        np.testing.assert_allclose(self.cube('cond_density_state_0.cube')['data'],ref,atol=2e-7,rtol=1e-5)

    def test_fail_closed_for_missing_imaginary_and_grid(self):
        Path('cube_001_eigenstate_imag_00001_spin_1_k_point_0001.cube').unlink()
        with self.assertRaisesRegex(ValueError,'exactly one cube'):
            self.menu(7,self.common+[1,2,*self.origin,'',0])
        content=b.load_cube_target('cube_*_eigenstate_density_{:05d}_spin_1_k_point_{:04d}.cube',1,1)
        content['origin']+=1
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            b.load_cube_target('cube_*_eigenstate_density_{:05d}_spin_1_k_point_{:04d}.cube',2,1,content)

    def test_metadata_dimensions_bands_grid_and_origin(self):
        with self.assertRaises(ValueError):b.get_bse_weights('states.npz',0,1,2,2)
        with self.assertRaises(ValueError):b.check_band_window('states.npz',2,3)
        with self.assertRaises(ValueError):b.get_k_coords(0,1,2,1,'states.npz')
        with self.assertRaises(ValueError):b.read_elsi_state('states.npz',-1)
        with self.assertRaises(ValueError):b.get_grid_val_at_coord(np.ones((3,4,5)),self.atoms,[-10,0,0],self.origin)
        np.testing.assert_allclose(b.get_k_coords(1,2,1,1,'states.npz'),[.5,0,0])

    def test_elsi_real_complex_sparse_and_truncated(self):
        for complex_type in (False,True):
            header=[170915,0,int(complex_type),3,0,3]+[0]*10
            values=np.array([.5,1.,-.5],dtype='<c16' if complex_type else '<f8')
            if complex_type:values*=1+2j
            payload=struct.pack('<16q',*header)+np.array([1,3,3],dtype='<i8').tobytes()+np.array([1,3,2],dtype='<i4').tobytes()+values.tobytes()
            Path('elsi').write_bytes(payload)
            np.testing.assert_array_equal(b.read_elsi_state('elsi',0).toarray().ravel(),[values[0],0,values[1]])
            self.assertEqual(b.read_elsi_state('elsi',1).nnz,0)
            np.testing.assert_array_equal(b.read_elsi_state('elsi',2).toarray().ravel(),[0,values[2],0])
            with self.assertRaises(ValueError):b.read_elsi_state('elsi',3)
            Path('elsi').write_bytes(payload[:-1])
            with self.assertRaises(ValueError):b.read_elsi_state('elsi',0)

    def test_bundled_water_legacy_reader(self):
        path=Path(b.__file__).parent/'Water/BSE_vecs'
        if not path.exists():self.skipTest('Optional bundled Water data absent')
        # This historical fixture contains duplicate sparse rows; preserve SciPy
        # additive CSC semantics instead of rejecting an existing user dataset.
        vec=b.read_elsi_state(path,1).toarray()
        self.assertEqual(vec.shape,(25,1))
        self.assertTrue(np.isfinite(vec).all())

    def test_import_energy_units_and_physical_metadata(self):
        Path('input').mkdir();Path('output').mkdir()
        Path('libbse.in').write_text('input_dir input\noutput_dir output\nnocc 1\nnvirt 2\nwavefunction_gauge native\nqp_data energy_qp\n')
        Path('input/bz_sampling_out').write_text('2 1 1\n2 2\n1 0.5 0 0 0\n2 0.5 0.5 0 0\n')
        Path('energy_qp').write_text('K_point 1 : 0 0 0\n1 2 -1 -1\n2 0 1 1\nK_point 2 : 0.5 0 0\n1 2 -1 -1\n2 0 1 1\n')
        Path('output/Excitation_Energy_singlet.dat').write_text('0.2\n')
        Path('output/oscillator_strength_singlet_tda.dat').write_text('0 2.7211386245988 0 0 0 0\n')
        Path('output/Excitation_Amplitude_singlet_0.dat').write_text(' '.join(f'({z.real},{z.imag})' for z in self.coeff.ravel())+'\n')
        convert('.', 'import.npz',1,2)
        with np.load('import.npz') as z:
            np.testing.assert_allclose(z['energies_ha'],[0.1])
            np.testing.assert_allclose(z['energies_ry'],[0.2])
            np.testing.assert_allclose(z['coefficients'],self.coeff)
        Path('input/bandmlk1001.out').write_text(
            self.mulliken_block(.5, .2)+self.mulliken_block(0, 0))
        result = convert('.', 'embedded.npz', 1, 2)
        self.assertEqual(result['mulliken_kpoints'], 2)
        with np.load('embedded.npz', allow_pickle=False) as z:
            np.testing.assert_allclose(z['mulliken_populations'][:,:,0,0],
                                       [[.1,.2,.3],[.3,.4,.5]])
            self.assertIn('bandmlk1001.out', str(z['provenance']))
        with self.assertRaisesRegex(ValueError, 'No bandmlk'):
            convert('.', 'missing.npz', 1, 2, mulliken_dir='absent')
        Path('output/oscillator_strength_singlet_tda.dat').write_text('0 5.4422772491976 0 0 0 0\n')
        with self.assertRaisesRegex(ValueError,'units/order'):convert('.', 'bad.npz',1,2)

    def test_unequal_rank_partition(self):
        vec=np.array([[1,0,0,0,0],[0,0,0,0,1]],complex)
        for rank,chunk in enumerate([vec[:,:3],vec[:,3:]]):
            Path(f'rank{rank}').write_text('\n'.join(' '.join(f'({z.real},{z.imag})' for z in row) for row in chunk)+'\n')
        np.testing.assert_array_equal(read_amplitudes(['rank0','rank1'],5,2),vec)
        with self.assertRaises(ValueError):read_amplitudes(['rank1','rank0'],5,2)

if __name__=='__main__':unittest.main(verbosity=2)
