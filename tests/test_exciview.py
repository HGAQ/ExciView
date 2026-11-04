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


    def mulliken_block(self, kx, shift):
        text = f'k point number: 1: ( {kx} 0 0 )\n'
        for band in (1, 2, 3):
            pop = .1*band + shift
            text += (f'State {band}\n'
                     f'{band} 0 1 1 {pop} {pop*.4} {pop*.6}\n'
                     f'{band} 0 1 2 {1-pop} {(1-pop)*.4} {(1-pop)*.6}\n')
        return text


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
