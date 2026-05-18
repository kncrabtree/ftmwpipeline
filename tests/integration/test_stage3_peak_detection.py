"""
Stage 3 integration tests on real experiment 2638 data.

* cross-interface consistency: CLI == Pipeline == functional API
* sanity vs. known strong lines
* the gap pass demonstrably recovers a weak line the windowed primary
  pass misses

Heavy (full 750k FID + repeated FT/noise); marked slow + integration.
"""

import subprocess

import numpy as np
import pytest

from ftmwpipeline import Pipeline
import ftmwpipeline.api as ftmw

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRIM = (26500.0, 40000.0)
# Strong lines observed in exp 2638 (MHz); detection must land on these.
KNOWN_STRONG = [28817.3, 31328.1, 33839.0, 36350.0, 38861.0]


def _prep(path, data_path):
    """Stage 0->2 on a fresh .ftmw via the functional API."""
    ftmw.import_data(path, source=data_path, force=True)
    ftmw.compute_ft(path, zpf=2, expf_us=5.0, trim=TRIM)
    ftmw.estimate_noise(path)


def _arr(peaks):
    f = np.array([p.frequency for p in peaks])
    s = np.array([p.snr for p in peaks])
    c = [p.classification.value if p.classification else "" for p in peaks]
    return f, s, c


def test_cross_interface_consistency(exp_2638_data_path, temp_ftmw_dir):
    pfile = temp_ftmw_dir / "p.ftmw"
    ffile = temp_ftmw_dir / "f.ftmw"
    cfile = temp_ftmw_dir / "c.ftmw"
    for fp in (pfile, ffile, cfile):
        _prep(fp, exp_2638_data_path)

    peaks_pipe = Pipeline(pfile).detect_peaks(min_snr=3.0, trim=TRIM)
    peaks_func = ftmw.detect_peaks(ffile, min_snr=3.0, trim=TRIM)

    res = subprocess.run(
        [
            "ftmwpipeline", "detect-peaks", str(cfile),
            "--trim", "26500:40000", "--min-snr", "3.0",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    peaks_cli = ftmw.load_peaks(cfile)

    fp_, sp_, cp_ = _arr(peaks_pipe)
    ff_, sf_, cf_ = _arr(peaks_func)
    fc_, sc_, cc_ = _arr(peaks_cli)

    assert len(peaks_pipe) == len(peaks_func) == len(peaks_cli)
    np.testing.assert_allclose(fp_, ff_, rtol=0, atol=1e-9)
    np.testing.assert_allclose(fp_, fc_, rtol=0, atol=1e-9)
    np.testing.assert_allclose(sp_, sf_, rtol=0, atol=1e-6)
    np.testing.assert_allclose(sp_, sc_, rtol=0, atol=1e-6)
    assert cp_ == cf_ == cc_


def test_detection_is_sane_vs_known_lines(
    exp_2638_data_path, temp_ftmw_dir
):
    fp = temp_ftmw_dir / "sane.ftmw"
    _prep(fp, exp_2638_data_path)
    peaks = ftmw.detect_peaks(fp, min_snr=3.0, trim=TRIM)

    assert 100 < len(peaks) < 5000, f"implausible peak count {len(peaks)}"
    freqs = np.array([p.frequency for p in peaks])
    # All peaks lie within the trim window.
    assert freqs.min() >= TRIM[0] - 1 and freqs.max() <= TRIM[1] + 1
    # Every known strong line has a detection within 1 MHz.
    for line in KNOWN_STRONG:
        assert np.min(np.abs(freqs - line)) < 1.0, (
            f"no detection near known line {line} MHz"
        )
    # The strongest detections should be classified strong.
    strong = [p for p in peaks if p.is_strong]
    assert strong, "no peaks classified strong"
    assert max(p.snr for p in peaks) == pytest.approx(
        max(p.snr for p in strong)
    )


def test_gap_pass_recovers_a_weak_line(exp_2638_data_path, temp_ftmw_dir):
    """With the gap pass on, at least one weak line is recovered in a region
    the apodized primary pass did not cover (provenance == 'gap', and outside
    every primary leakage exclusion)."""
    fp = temp_ftmw_dir / "gap.ftmw"
    _prep(fp, exp_2638_data_path)

    no_gap = ftmw.detect_peaks(
        fp, min_snr=3.0, trim=TRIM, run_gap_pass=False
    )
    with_gap = ftmw.detect_peaks(fp, min_snr=3.0, trim=TRIM)

    assert all(
        p.properties["detection_pass"] == "primary" for p in no_gap
    )
    gap_peaks = [
        p for p in with_gap if p.properties["detection_pass"] == "gap"
    ]
    assert gap_peaks, "gap pass recovered nothing"
    assert len(with_gap) > len(no_gap)

    # A recovered gap peak must not coincide with any primary detection
    # (it is genuinely in a gap, not a re-find of a primary line).
    primary_freqs = np.array(
        [p.frequency for p in with_gap
         if p.properties["detection_pass"] == "primary"]
    )
    recovered = [
        gp for gp in gap_peaks
        if np.min(np.abs(primary_freqs - gp.frequency)) > 0.5
    ]
    assert recovered, "no gap peak is genuinely separated from primaries"
