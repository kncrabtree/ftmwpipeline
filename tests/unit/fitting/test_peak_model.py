"""
Unit tests for the Stage 5 finite-acquisition line-shape model.

Covers :mod:`ftmwpipeline.fitting.peak_model`: the closed-form ``h_T`` against
a literal numerical FFT, its analytic Jacobian against finite differences, the
demodulation / sideband mapping on synthetic lines of *both* sidebands (a
wrong sign must be caught -- it is a silent 100s-of-kHz frequency bias), and
the de-ramp round trip. Conventions and tolerances follow the Stage 5
method doc (``docs/source/methods/stage5_fitting.rst``).
"""

import numpy as np
import pytest
from scipy.optimize import least_squares

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    PeakShape,
    baseband_offset,
    effective_tau,
    effective_tau_gaussian,
    effective_tau_shape,
    h_T,
    h_T_gaussian,
    h_T_gaussian_jacobian,
    h_T_jacobian,
    h_T_shape,
    h_T_shape_jacobian,
    model_spectrum,
    molecular_frequency,
    sideband_sign,
    to_baseband_offset,
)
from ftmwpipeline.preprocessing.leakage import deramp_to_active_start

# Physical scale of the 2638 fixture (see prototype.py).
T_US = 12.65  # active acquisition length
TAU_US = 5.0  # effective decay constant (expf_us = 5 dominates)
DF_MHZ = 0.0122  # FT bin spacing (~12 kHz)
PROBE_MHZ = 40960.0
START_US = 2.35  # active-region turn-on (15 us record - 12.65 us active)


def _offset_grid(half_width_mhz: float = 2.0, df_mhz: float = DF_MHZ) -> np.ndarray:
    """Symmetric baseband-offset grid at the FT bin spacing."""
    n = int(round(half_width_mhz / df_mhz))
    return np.arange(-n, n + 1) * df_mhz


def _numerical_fft_response(
    delta_f_mhz: np.ndarray,
    tau_us: float,
    acquisition_us: float,
    fs_mhz: float = 50.0,
) -> np.ndarray:
    """Literal numerical realization of the model (prototype cross-check).

    Synthesizes a damped-cosine FID over ``[0, T]``, rfft's the zero-padded
    record, and interpolates the response near the line onto ``delta_f_mhz``.
    Confirms ``h_T`` *is* the FFT of the finite-T damped cosine.
    """
    dt_us = 1.0 / fs_mhz
    n_active = int(round(acquisition_us / dt_us))
    f0_mhz = 5.0  # park the line well inside Nyquist
    n_pad = 1 << (int(np.log2(n_active)) + 4)
    t = np.arange(n_active) * dt_us
    fid = np.cos(2.0 * np.pi * f0_mhz * t) * np.exp(-t / tau_us)
    rec = np.zeros(n_pad)
    rec[:n_active] = fid
    spec = np.fft.rfft(rec) * dt_us  # -> continuous-transform units (µs)
    freqs_mhz = np.fft.rfftfreq(n_pad, d=dt_us)
    offs_mhz = freqs_mhz - f0_mhz
    keep = np.abs(offs_mhz) <= float(np.max(np.abs(delta_f_mhz))) + 1.0
    re = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).real)
    im = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).imag)
    return re + 1j * im


# ---------------------------------------------------------------------------
# h_T -- the finite-T line shape
# ---------------------------------------------------------------------------
class TestFiniteTResponse:
    def test_closed_form_matches_numerical_fft(self):
        """h_T equals a literal numerical FFT of a synthesized damped cosine."""
        u = _offset_grid(2.0)
        analytic = h_T(u, TAU_US, T_US)
        numeric = _numerical_fft_response(u, TAU_US, T_US)
        rel_err = np.abs(analytic - numeric) / np.abs(analytic).max()
        # The residual is the numerical FFT's finite-grid interpolation error;
        # the prototype measured ~7e-3. The closed form itself is exact.
        assert rel_err.max() < 1.5e-2

    def test_on_line_value_is_effective_tau(self):
        """h_T(0) is real and equals tau_eff = tau(1 - e^{-T/tau})."""
        on_line = h_T(np.array([0.0]), TAU_US, T_US)[0]
        assert on_line.real == pytest.approx(effective_tau(TAU_US, T_US))
        assert on_line.imag == pytest.approx(0.0, abs=1e-12)

    def test_conjugate_symmetry_in_offset(self):
        """h_T(-Δf) = conj(h_T(Δf)) exactly -- the load-bearing sign property."""
        u = _offset_grid(2.0)
        assert np.allclose(h_T(-u, TAU_US, T_US), np.conj(h_T(u, TAU_US, T_US)))

    def test_skirt_decays_as_inverse_offset(self):
        """Far from center the magnitude follows the 1/|Δf| leakage envelope."""
        far = np.array([1.0, 2.0, 4.0])  # MHz, well outside the core
        mag = np.abs(h_T(far, TAU_US, T_US))
        # |h_T| ~ const / |Δf|: the product |Δf|·|h_T| is roughly flat.
        product = far * mag
        assert np.allclose(product, product[0], rtol=0.05)

    def test_undamped_limit_tends_to_acquisition_length(self):
        """As tau -> infinity, h_T(0) -> T (the boxcar limit)."""
        on_line = h_T(np.array([0.0]), 1.0e9, T_US)[0]
        assert on_line.real == pytest.approx(T_US, rel=1e-4)

    def test_accepts_scalar_input(self):
        """A scalar offset is accepted and yields a 0-d complex array."""
        out = h_T(0.0, TAU_US, T_US)
        assert np.iscomplexobj(out)
        assert float(out.real) == pytest.approx(effective_tau(TAU_US, T_US))

    @pytest.mark.parametrize("tau,acq", [(0.0, T_US), (-1.0, T_US), (TAU_US, 0.0)])
    def test_rejects_non_positive_parameters(self, tau, acq):
        with pytest.raises(ValueError):
            h_T(_offset_grid(0.5), tau, acq)


# ---------------------------------------------------------------------------
# h_T_jacobian -- analytic derivatives
# ---------------------------------------------------------------------------
class TestJacobian:
    def test_d_delta_f_matches_finite_difference(self):
        """Analytic dh/d(Δf) matches a central finite difference."""
        u = _offset_grid(2.0)
        dh_ddf, _ = h_T_jacobian(u, TAU_US, T_US)
        eps = 1e-6
        fd = (h_T(u + eps, TAU_US, T_US) - h_T(u - eps, TAU_US, T_US)) / (2 * eps)
        rel_err = np.abs(dh_ddf - fd).max() / np.abs(dh_ddf).max()
        assert rel_err < 1e-6

    def test_d_tau_matches_finite_difference(self):
        """Analytic dh/d(τ) matches a central finite difference."""
        u = _offset_grid(2.0)
        _, dh_dtau = h_T_jacobian(u, TAU_US, T_US)
        eps = 1e-5
        fd = (h_T(u, TAU_US + eps, T_US) - h_T(u, TAU_US - eps, T_US)) / (2 * eps)
        rel_err = np.abs(dh_dtau - fd).max() / np.abs(dh_dtau).max()
        assert rel_err < 1e-6

    @pytest.mark.parametrize("tau,acq", [(0.0, T_US), (TAU_US, -1.0)])
    def test_rejects_non_positive_parameters(self, tau, acq):
        with pytest.raises(ValueError):
            h_T_jacobian(_offset_grid(0.5), tau, acq)


# ---------------------------------------------------------------------------
# effective_tau
# ---------------------------------------------------------------------------
class TestEffectiveTau:
    def test_closed_form(self):
        expected = TAU_US * (1.0 - np.exp(-T_US / TAU_US))
        assert effective_tau(TAU_US, T_US) == pytest.approx(expected)

    def test_undamped_limit(self):
        assert effective_tau(1.0e9, T_US) == pytest.approx(T_US, rel=1e-4)

    @pytest.mark.parametrize("tau,acq", [(0.0, T_US), (TAU_US, 0.0)])
    def test_rejects_non_positive_parameters(self, tau, acq):
        with pytest.raises(ValueError):
            effective_tau(tau, acq)


# ---------------------------------------------------------------------------
# Gaussian envelope -- h_T_gaussian, Jacobian, effective_tau, dispatcher
# ---------------------------------------------------------------------------
TAU_G_US = 8.0  # representative τ_G from the Voigt-deficit Part B calibration


def _numerical_fft_gaussian(
    delta_f_mhz: np.ndarray,
    tau_G_us: float,
    acquisition_us: float,
    fs_mhz: float = 50.0,
) -> np.ndarray:
    """Numerical FFT of a Gaussian-windowed damped cosine -- cross-check.

    Synthesizes ``exp(-(t/τ_G)²) cos(2π f₀ t)`` on ``[0, T]``, rfft's the
    zero-padded record, and interpolates onto ``delta_f_mhz``. Confirms
    :func:`h_T_gaussian` *is* the FT of the finite-T Gaussian-windowed cosine.
    """
    dt_us = 1.0 / fs_mhz
    n_active = int(round(acquisition_us / dt_us))
    f0_mhz = 5.0
    n_pad = 1 << (int(np.log2(n_active)) + 4)
    t = np.arange(n_active) * dt_us
    fid = np.cos(2.0 * np.pi * f0_mhz * t) * np.exp(-((t / tau_G_us) ** 2))
    rec = np.zeros(n_pad)
    rec[:n_active] = fid
    spec = np.fft.rfft(rec) * dt_us
    freqs_mhz = np.fft.rfftfreq(n_pad, d=dt_us)
    offs_mhz = freqs_mhz - f0_mhz
    keep = np.abs(offs_mhz) <= float(np.max(np.abs(delta_f_mhz))) + 1.0
    re = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).real)
    im = np.interp(delta_f_mhz, offs_mhz[keep], (2.0 * spec[keep]).imag)
    return re + 1j * im


class TestHTGaussian:
    def test_center_is_effective_tau(self):
        """``h_T_gaussian(0; τ_G, T) = effective_tau_gaussian(τ_G, T)``, real."""
        z = h_T_gaussian(np.array([0.0]), TAU_G_US, T_US)
        assert z.imag[0] == pytest.approx(0.0, abs=1e-12)
        assert z.real[0] == pytest.approx(
            effective_tau_gaussian(TAU_G_US, T_US),
            rel=1e-12,
        )

    def test_matches_numerical_fft(self):
        """Closed-form matches a literal rfft of the Gaussian-windowed cosine."""
        u = _offset_grid(2.0)
        analytic = h_T_gaussian(u, TAU_G_US, T_US)
        numerical = _numerical_fft_gaussian(u, TAU_G_US, T_US)
        rel_err = np.abs(analytic - numerical) / np.abs(analytic).max()
        # Same tolerance convention as TestFiniteTResponse: residual is the
        # numerical FFT's finite-grid interpolation error, not the closed
        # form. The closed form is verified to machine precision below
        # against scipy.integrate.quad.
        assert rel_err.max() < 1.5e-2

    def test_matches_quadrature_to_machine_precision(self):
        """Closed form matches direct quadrature of the time-domain integral."""
        from scipy.integrate import quad as scipy_quad

        def _quad_h(df: float, tau_G: float, T: float) -> complex:
            re, _ = scipy_quad(
                lambda t: np.exp(-((t / tau_G) ** 2)) * np.cos(2 * np.pi * df * t),
                0,
                T,
                epsabs=1e-14,
                epsrel=1e-14,
            )
            im, _ = scipy_quad(
                lambda t: -np.exp(-((t / tau_G) ** 2)) * np.sin(2 * np.pi * df * t),
                0,
                T,
                epsabs=1e-14,
                epsrel=1e-14,
            )
            return re + 1j * im

        for df in [0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0]:
            analytic = h_T_gaussian(np.array([df]), TAU_G_US, T_US)[0]
            quad_val = _quad_h(df, TAU_G_US, T_US)
            scale = max(abs(quad_val), 1e-12)
            assert (
                abs(analytic - quad_val) / scale < 1e-12
            ), f"Δf={df}: analytic={analytic} vs quad={quad_val}"

    def test_hermitian_symmetry(self):
        """``h_T_gaussian(-Δf) = conj(h_T_gaussian(Δf))`` exactly."""
        u = _offset_grid(2.0)
        pos = h_T_gaussian(u, TAU_G_US, T_US)
        neg = h_T_gaussian(-u, TAU_G_US, T_US)
        rel_err = np.abs(neg - np.conj(pos)).max() / np.abs(pos).max()
        assert rel_err < 1e-12

    def test_long_tau_limit_approaches_boxcar(self):
        """For τ_G >> T, ``h_T_gaussian(0) → T`` (full boxcar integral)."""
        big = h_T_gaussian(np.array([0.0]), 1e6, T_US)
        assert big.real[0] == pytest.approx(T_US, rel=1e-4)

    @pytest.mark.parametrize("tau_G,acq", [(0.0, T_US), (-1.0, T_US), (TAU_G_US, 0.0)])
    def test_rejects_non_positive_parameters(self, tau_G, acq):
        with pytest.raises(ValueError):
            h_T_gaussian(_offset_grid(0.5), tau_G, acq)


class TestHTGaussianJacobian:
    def test_d_delta_f_matches_finite_difference(self):
        u = _offset_grid(2.0)
        dh_ddf, _ = h_T_gaussian_jacobian(u, TAU_G_US, T_US)
        eps = 1e-6
        fd = (
            h_T_gaussian(u + eps, TAU_G_US, T_US)
            - h_T_gaussian(u - eps, TAU_G_US, T_US)
        ) / (2 * eps)
        rel_err = np.abs(dh_ddf - fd).max() / np.abs(dh_ddf).max()
        assert rel_err < 1e-6

    def test_d_tau_matches_finite_difference(self):
        u = _offset_grid(2.0)
        _, dh_dtau = h_T_gaussian_jacobian(u, TAU_G_US, T_US)
        eps = 1e-5
        fd = (
            h_T_gaussian(u, TAU_G_US + eps, T_US)
            - h_T_gaussian(u, TAU_G_US - eps, T_US)
        ) / (2 * eps)
        rel_err = np.abs(dh_dtau - fd).max() / np.abs(dh_dtau).max()
        assert rel_err < 1e-6

    @pytest.mark.parametrize("tau_G,acq", [(0.0, T_US), (TAU_G_US, -1.0)])
    def test_rejects_non_positive_parameters(self, tau_G, acq):
        with pytest.raises(ValueError):
            h_T_gaussian_jacobian(_offset_grid(0.5), tau_G, acq)


class TestEffectiveTauGaussian:
    def test_closed_form(self):
        from scipy.special import erf

        expected = TAU_G_US * np.sqrt(np.pi) / 2.0 * erf(T_US / TAU_G_US)
        assert effective_tau_gaussian(TAU_G_US, T_US) == pytest.approx(
            expected,
            rel=1e-12,
        )

    def test_undamped_limit(self):
        # τ_G → ∞: full boxcar integral = T.
        assert effective_tau_gaussian(1e9, T_US) == pytest.approx(T_US, rel=1e-4)

    @pytest.mark.parametrize("tau_G,acq", [(0.0, T_US), (TAU_G_US, 0.0)])
    def test_rejects_non_positive_parameters(self, tau_G, acq):
        with pytest.raises(ValueError):
            effective_tau_gaussian(tau_G, acq)


class TestPeakShapeDispatchers:
    def test_lorentzian_path(self):
        u = _offset_grid(1.0)
        assert np.array_equal(
            h_T_shape(PeakShape.LORENTZIAN, u, TAU_US, T_US),
            h_T(u, TAU_US, T_US),
        )
        a1, b1 = h_T_shape_jacobian(PeakShape.LORENTZIAN, u, TAU_US, T_US)
        a2, b2 = h_T_jacobian(u, TAU_US, T_US)
        assert np.array_equal(a1, a2)
        assert np.array_equal(b1, b2)
        assert effective_tau_shape(PeakShape.LORENTZIAN, TAU_US, T_US) == pytest.approx(
            effective_tau(TAU_US, T_US)
        )

    def test_gaussian_path(self):
        u = _offset_grid(1.0)
        assert np.array_equal(
            h_T_shape(PeakShape.GAUSSIAN, u, TAU_G_US, T_US),
            h_T_gaussian(u, TAU_G_US, T_US),
        )
        a1, b1 = h_T_shape_jacobian(PeakShape.GAUSSIAN, u, TAU_G_US, T_US)
        a2, b2 = h_T_gaussian_jacobian(u, TAU_G_US, T_US)
        assert np.array_equal(a1, a2)
        assert np.array_equal(b1, b2)
        assert effective_tau_shape(PeakShape.GAUSSIAN, TAU_G_US, T_US) == pytest.approx(
            effective_tau_gaussian(TAU_G_US, T_US)
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("lorentzian", PeakShape.LORENTZIAN),
            ("Lorentzian", PeakShape.LORENTZIAN),
            ("gaussian", PeakShape.GAUSSIAN),
            ("GAUSSIAN", PeakShape.GAUSSIAN),
            (PeakShape.GAUSSIAN, PeakShape.GAUSSIAN),
        ],
    )
    def test_coerce_accepts(self, value, expected):
        assert PeakShape.coerce(value) is expected

    @pytest.mark.parametrize("value", ["voigt", "exp", "", None, 42])
    def test_coerce_rejects(self, value):
        with pytest.raises(ValueError):
            PeakShape.coerce(value)


# ---------------------------------------------------------------------------
# The demodulation / sideband mapping
# ---------------------------------------------------------------------------
class TestSidebandSign:
    def test_lower_is_negative(self):
        assert sideband_sign(Sideband.LOWER) == -1.0
        assert sideband_sign("lower") == -1.0
        assert sideband_sign("LSB") == -1.0

    def test_upper_is_positive(self):
        assert sideband_sign(Sideband.UPPER) == 1.0
        assert sideband_sign("upper") == 1.0
        assert sideband_sign("USB") == 1.0

    def test_rejects_unknown_string(self):
        with pytest.raises(ValueError):
            sideband_sign("middle")


class TestBasebandMapping:
    @pytest.mark.parametrize("sideband", ["lower", "upper"])
    def test_round_trip(self, sideband):
        """molecular_frequency inverts baseband_offset exactly."""
        f_c = PROBE_MHZ + sideband_sign(sideband) * 8.0
        f = f_c + np.linspace(-2.0, 2.0, 51)
        u = baseband_offset(f, f_c, sideband)
        assert np.allclose(molecular_frequency(u, f_c, sideband), f)

    def test_lower_sideband_descends(self):
        """Lower sideband: rising baseband offset maps to falling molecular f."""
        f_c = 40952.0
        u = baseband_offset(np.array([f_c - 1.0, f_c, f_c + 1.0]), f_c, "lower")
        assert np.allclose(u, [1.0, 0.0, -1.0])

    def test_upper_sideband_ascends(self):
        """Upper sideband: baseband offset and molecular offset share sign."""
        f_c = 40968.0
        u = baseband_offset(np.array([f_c - 1.0, f_c, f_c + 1.0]), f_c, "upper")
        assert np.allclose(u, [-1.0, 0.0, 1.0])

    def test_center_maps_to_zero_offset(self):
        for sb in ("lower", "upper"):
            assert baseband_offset(np.array([40955.0]), 40955.0, sb)[0] == 0.0


# ---------------------------------------------------------------------------
# ModelPeak.peak_uid -- the optional identity slot (P2)
# ---------------------------------------------------------------------------
class TestModelPeakIdentitySlot:
    def test_defaults_to_none(self):
        """None of the ~22 existing construction sites pass peak_uid; the
        field must default so they keep working, and the default must read
        as "unstamped", not a fabricated identity."""
        peak = ModelPeak(amplitude=2.0, offset_mhz=0.31, phase=1.1)
        assert peak.peak_uid is None

    def test_positional_construction_still_works(self):
        """peak_uid is trailing, so ModelPeak(amp, off, phase) positional
        construction (used throughout the fitting modules) is unaffected."""
        peak = ModelPeak(1.0, 0.1, 0.3)
        assert peak.peak_uid is None

    def test_can_be_stamped(self):
        peak = ModelPeak(amplitude=1.0, offset_mhz=0.0, phase=0.0, peak_uid=12345)
        assert peak.peak_uid == 12345

    def test_stamped_peak_still_models_identically(self):
        """peak_uid plays no role in the spectrum model -- stamping a peak
        must not move any fitted number."""
        u = _offset_grid(1.0)
        bare = ModelPeak(amplitude=3.0, offset_mhz=0.4, phase=0.7)
        stamped = ModelPeak(amplitude=3.0, offset_mhz=0.4, phase=0.7, peak_uid=999)
        assert np.array_equal(
            model_spectrum(u, [bare], TAU_US, T_US),
            model_spectrum(u, [stamped], TAU_US, T_US),
        )


# ---------------------------------------------------------------------------
# model_spectrum -- the window model
# ---------------------------------------------------------------------------
class TestModelSpectrum:
    def test_empty_peak_list_is_all_zero(self):
        u = _offset_grid(1.0)
        out = model_spectrum(u, [], TAU_US, T_US)
        assert out.shape == u.shape
        assert np.iscomplexobj(out)
        assert np.all(out == 0.0)

    def test_superposition(self):
        """The model is linear: a two-line model is the sum of one-line models."""
        u = _offset_grid(2.5)
        p1 = ModelPeak(amplitude=3.0, offset_mhz=-0.4, phase=0.7)
        p2 = ModelPeak(amplitude=1.5, offset_mhz=0.6, phase=2.3)
        both = model_spectrum(u, [p1, p2], TAU_US, T_US)
        sep = model_spectrum(u, [p1], TAU_US, T_US) + model_spectrum(
            u, [p2], TAU_US, T_US
        )
        assert np.allclose(both, sep)

    def test_on_line_value(self):
        """A single line evaluated at its own offset gives ½ A e^{iφ} τ_eff."""
        peak = ModelPeak(amplitude=2.0, offset_mhz=0.31, phase=1.1)
        out = model_spectrum(np.array([peak.offset_mhz]), [peak], TAU_US, T_US)[0]
        expected = (
            0.5 * peak.amplitude * np.exp(1j * peak.phase) * effective_tau(TAU_US, T_US)
        )
        assert out == pytest.approx(expected)

    @pytest.mark.parametrize("tau,acq", [(0.0, T_US), (TAU_US, -1.0)])
    def test_rejects_non_positive_parameters(self, tau, acq):
        with pytest.raises(ValueError):
            model_spectrum(_offset_grid(0.5), [], tau, acq)


# ---------------------------------------------------------------------------
# Grid conversion: molecular frequency -> signed baseband offset
# ---------------------------------------------------------------------------
class TestToBasebandOffset:
    """``to_baseband_offset`` is grid relabel only (D9: no deramp).

    The active-FT (:mod:`ftmwpipeline.fitting.active_ft`) is in the ``[0, T]``
    form natively, so Stage 5 no longer applies a phase-ramp correction at
    window-materialization time. ``deramp_to_active_start`` is still exercised
    in ``tests/unit/preprocessing/test_leakage.py`` as a stand-alone primitive
    used by Stage 4's edge-coherence work.
    """

    @pytest.mark.parametrize("sideband", ["lower", "upper"])
    def test_grid_conversion_on_both_sidebands(self, sideband):
        s = sideband_sign(sideband)
        f_c = PROBE_MHZ + s * 8.0
        u = _offset_grid(2.0)
        peaks = [
            ModelPeak(amplitude=4.0, offset_mhz=-0.3, phase=0.5),
            ModelPeak(amplitude=2.0, offset_mhz=0.7, phase=2.0),
        ]
        x0 = model_spectrum(u, peaks, TAU_US, T_US)  # [0, T] frame
        f_grid = molecular_frequency(u, f_c, sideband)

        u_out, z_out = to_baseband_offset(
            f_grid,
            x0,
            center_mhz=f_c,
            sideband=sideband,
        )
        assert np.allclose(u_out, u)
        # Grid relabel only -- the spectrum is returned unchanged.
        assert np.allclose(z_out, x0)

    def test_descending_grid_preserved(self):
        """Lower-sideband molecular grid descends; conversion preserves order."""
        f_c = 40952.0
        u = _offset_grid(1.0)
        f_grid = molecular_frequency(u, f_c, "lower")  # descending in MHz
        z = model_spectrum(u, [ModelPeak(1.0, 0.1, 0.3)], TAU_US, T_US)
        u_out, z_out = to_baseband_offset(f_grid, z, center_mhz=f_c, sideband="lower")
        assert np.allclose(u_out, u)
        assert np.allclose(z_out, z)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal shape"):
            to_baseband_offset(
                np.zeros(8),
                np.zeros(7, dtype=np.complex128),
                center_mhz=0.0,
                sideband="lower",
            )


class TestDeRampPrimitive:
    """``deramp_to_active_start`` survives as the Stage 4 leakage primitive.

    These checks live here for the few callers in the fitting tests that used
    to compose deramp + grid conversion via the now-removed ``to_baseband_frame``
    helper; the comprehensive deramp coverage is in
    ``tests/unit/preprocessing/test_leakage.py``.
    """

    def test_deramp_preserves_magnitude(self):
        """De-ramping is a per-bin phase rotation; magnitudes are unchanged."""
        f = 40952.0 + _offset_grid(1.0)
        z = model_spectrum(_offset_grid(1.0), [ModelPeak(2.0, 0.0, 1.0)], TAU_US, T_US)
        deramped = deramp_to_active_start(f, z, PROBE_MHZ, START_US)
        assert np.allclose(np.abs(deramped), np.abs(z))


# ---------------------------------------------------------------------------
# The sideband mapping on synthetic lines -- a wrong sign must be caught
# ---------------------------------------------------------------------------
def _fit_single_line(
    u: np.ndarray, z: np.ndarray, tau_us: float, acquisition_us: float
) -> tuple[ModelPeak, float]:
    """Least-squares recover one line's (A, offset, phase) from window data.

    A minimal test-local solver (the production window-fit core is Stage 5
    task 3); it exists only to exercise the sideband mapping. Returns the
    fitted peak and the relative RMS residual ``||z - model|| / ||z||``.
    """
    mag = np.abs(z)
    i0 = int(np.argmax(mag))
    a0 = 2.0 * mag[i0] / effective_tau(tau_us, acquisition_us)
    init = np.array([a0, float(u[i0]), float(np.angle(z[i0]))])
    lo = np.array([0.0, float(u.min()), -4.0 * np.pi])
    hi = np.array([np.inf, float(u.max()), 4.0 * np.pi])

    def residual(p: np.ndarray) -> np.ndarray:
        peak = ModelPeak(amplitude=p[0], offset_mhz=p[1], phase=p[2])
        r = z - model_spectrum(u, [peak], tau_us, acquisition_us)
        return np.concatenate([r.real, r.imag])

    sol = least_squares(residual, np.clip(init, lo, hi), bounds=(lo, hi))
    peak = ModelPeak(amplitude=sol.x[0], offset_mhz=sol.x[1], phase=sol.x[2])
    model = model_spectrum(u, [peak], tau_us, acquisition_us)
    rel_resid = float(np.sqrt(np.sum(np.abs(z - model) ** 2) / np.sum(np.abs(z) ** 2)))
    return peak, rel_resid


class TestSidebandRecovery:
    """Synthetic single lines on both sidebands; a wrong sign must be caught."""

    @pytest.mark.parametrize("sideband", ["lower", "upper"])
    def test_correct_sign_recovers_frequency(self, sideband):
        """The correct sideband sign recovers the line and fits exactly."""
        s = sideband_sign(sideband)
        f_c = PROBE_MHZ + s * 8.0
        true_offset, true_phase, amp = 0.35, 1.1, 30.0
        f_line = molecular_frequency(np.array([true_offset]), f_c, sideband)[0]

        # Synthesize the de-ramped window data on its molecular grid.
        u_true = _offset_grid(2.0)
        f_grid = molecular_frequency(u_true, f_c, sideband)
        true_peak = ModelPeak(amplitude=amp, offset_mhz=true_offset, phase=true_phase)
        z = model_spectrum(u_true, [true_peak], TAU_US, T_US)

        # The fitter sees the molecular grid converted with the CORRECT sign.
        u_fit = baseband_offset(f_grid, f_c, sideband)
        order = np.argsort(u_fit)
        fit, rel_resid = _fit_single_line(u_fit[order], z[order], TAU_US, T_US)
        f_fit = molecular_frequency(np.array([fit.offset_mhz]), f_c, sideband)[0]
        assert abs(f_fit - f_line) < 1.0e-3  # < 1 kHz on clean data
        assert rel_resid < 1.0e-4  # the model fits clean data essentially exactly

    @pytest.mark.parametrize("sideband", ["lower", "upper"])
    def test_wrong_sign_is_caught(self, sideband):
        """The wrong sideband sign conjugates the model -> an unfittable misfit.

        ``h_T(-Δf) = conj(h_T(Δf))``, so demodulating with the flipped sign
        asks the fitter to match a *conjugated* (offset-reflected) line shape
        with a non-reflected ``h_T``. No (A, offset, phase) can do that: the
        best fit leaves a large residual where the correct sign fits exactly.
        That residual is the signal that catches the silent sign error.
        """
        s = sideband_sign(sideband)
        wrong = "upper" if sideband == "lower" else "lower"
        f_c = PROBE_MHZ + s * 8.0
        true_offset, true_phase, amp = 0.35, 1.1, 30.0

        u_true = _offset_grid(2.0)
        f_grid = molecular_frequency(u_true, f_c, sideband)
        true_peak = ModelPeak(amplitude=amp, offset_mhz=true_offset, phase=true_phase)
        z = model_spectrum(u_true, [true_peak], TAU_US, T_US)

        # The fitter sees the grid converted with the WRONG sign.
        u_fit = baseband_offset(f_grid, f_c, wrong)
        order = np.argsort(u_fit)
        _, rel_resid = _fit_single_line(u_fit[order], z[order], TAU_US, T_US)
        assert rel_resid > 0.1  # the conjugated shape cannot be fit away
