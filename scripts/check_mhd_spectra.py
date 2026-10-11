"""Analytic and structural tests for the MHD spectral diagnostics."""
import unittest
import numpy as np
from mhd_spectra import (divergence, energies, error_norms, face_divergence,
                         mean_square, physical_fields, spectrum)


class SpectraTests(unittest.TestCase):
    def setUp(self):
        self.n = 32
        self.x, self.y = np.meshgrid((np.arange(self.n)+.5)/self.n,
                                     (np.arange(self.n)+.5)/self.n)
        self.rng = np.random.default_rng(817)

    def test_known_sinusoid_normalization(self):
        q = 3 * np.sin(2*np.pi*5*self.x)
        p = spectrum(q)
        self.assertAlmostEqual(p[5], 4.5)
        self.assertAlmostEqual(p.sum(), 4.5)

    def test_parseval_dc_vectors_and_corner_modes(self):
        for shape in ((32,32),(32,32,3)):
            q = 2 + self.rng.normal(size=shape)
            self.assertAlmostEqual(spectrum(q).sum(), mean_square(q))
        q = np.cos(2*np.pi*(14*self.x+14*self.y))
        self.assertGreater(spectrum(q)[19], .49)  # outside complete-annulus plotting range
        self.assertAlmostEqual(spectrum(q).sum(), .5)

    def test_equal_power_does_not_imply_equal_fields(self):
        a = np.sin(2*np.pi*3*self.x)
        b = np.cos(2*np.pi*3*self.x)
        np.testing.assert_allclose(spectrum(a), spectrum(b), atol=1e-15)
        self.assertAlmostEqual(spectrum(a-b).sum(), 1)

    def test_vector_l2_matches_error_spectrum_and_linf(self):
        ref = self.rng.normal(size=(32,32,3))
        actual = ref + self.rng.normal(size=ref.shape)
        norm = error_norms(actual, ref)
        self.assertAlmostEqual(spectrum(actual-ref).sum(), norm['absolute_L2']**2)
        self.assertAlmostEqual(norm['relative_Linf'], np.linalg.norm(actual-ref,axis=-1).max()/np.linalg.norm(ref,axis=-1).max())

    def test_analytic_divergence_and_central_operator(self):
        b = np.stack((np.sin(2*np.pi*3*self.x), np.cos(2*np.pi*2*self.y), np.zeros_like(self.x)),axis=-1)
        exact = 6*np.pi*np.cos(6*np.pi*self.x)-4*np.pi*np.sin(4*np.pi*self.y)
        np.testing.assert_allclose(divergence(b),exact,atol=2e-13)
        fd = self.n/2*(np.roll(b[...,0],-1,axis=1)-np.roll(b[...,0],1,axis=1)+np.roll(b[...,1],-1,axis=0)-np.roll(b[...,1],1,axis=0))
        np.testing.assert_allclose(divergence(b,'centered'),fd,atol=3e-14)

    def test_ct_curl_is_divergence_free_but_centered_fft_need_not_be(self):
        a = self.rng.normal(size=(32,32))
        bx = self.n*(np.roll(a,-1,axis=0)-a)
        by = -self.n*(np.roll(a,-1,axis=1)-a)
        np.testing.assert_allclose(face_divergence(bx,by),0,atol=3e-12)
        centered = np.stack(((bx+np.roll(bx,-1,axis=1))/2,
                             (by+np.roll(by,-1,axis=0))/2,np.zeros_like(a)),axis=-1)
        self.assertGreater(mean_square(divergence(centered)),1)

    def test_nyquist_derivative_is_zero(self):
        bx = np.tile((-1.)**np.arange(32),(32,1))
        b = np.stack((bx,np.zeros_like(bx),np.zeros_like(bx)),axis=-1)
        np.testing.assert_allclose(divergence(b),0,atol=1e-14)

    def test_energy_spectra_recover_resolved_energies(self):
        rho = .5+self.rng.random((32,32))
        mom = self.rng.normal(size=(32,32,3))
        b = self.rng.normal(size=(32,32,3))
        e = np.sum(mom*mom,axis=-1)/(2*rho)+np.sum(b*b,axis=-1)/2+2
        state = dict(density=rho,momentum=mom,magnetic=b,energy=e)
        values=energies(state)
        for name, (q,factor) in physical_fields(state).items():
            if name != 'density_fluctuation':
                self.assertAlmostEqual(spectrum(q,factor).sum(),values[name.removesuffix('_energy')])
        self.assertAlmostEqual(values['thermal'],2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
