"""The grain organ's referee: a print's noise must BE counting statistics.

The frame is in the tree, page-pinned: Ware's *Argyronomicon*
Appendix 21 (pp. 205-207) - the treatment he credits to Dr. A. E.
Saunders of Kodak Ltd. Density is Nutting coverage, D = kappa*N*a/A;
the count in an aperture fluctuates as sigma_N = sqrt(N); therefore

    sigma_D = sqrt(kappa * a * D / A)

and Selwyn's invariant sigma_D*sqrt(2A) does not depend on the
aperture. These tests hold the organ to that law THROUGH THE PRINTS -
uniform patches developed by process_print itself, read back by the
granularity module's densitometer - with a particle area INJECTED per
test, so the law is refereed independently of which processes carry
sourced particle constants (those are the dossier's, tested
separately).

No dial exists anywhere in this: the only free parameter is the
physical particle area, and the mean is preserved exactly because the
count is converted to density through the same Nutting relation it
was drawn from.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import granularity as G
from atlas_film.processes import KAPPA, PROCESSES, process_print

A_TEST = 1.0        # um^2, an injected micron-scale particle
PITCH = 25.0        # um, a 35mm-ish pixel


@pytest.fixture
def grained_silver(monkeypatch):
    monkeypatch.setitem(PROCESSES["silver"], "grain_um2", A_TEST)


@pytest.fixture
def grained_vandyke(monkeypatch):
    monkeypatch.setitem(PROCESSES["vandyke"], "grain_um2", A_TEST)


def test_an_ungrained_print_is_noiseless():
    """The default: no grain requested, none applied - a uniform
    negative develops to a uniform print, which is what keeps every
    golden row bit-identical while the organ exists. The bound is the
    reader's own floor, not zero: np.std of 4096 bit-identical values
    returns ~2e-16 of pairwise-summation dust, and demanding 0.0
    exactly would be asserting an accumulator, not a print."""
    img = G.uniform_print("silver", 1.0, shape=(64, 64))
    d = G.density_field("silver", img)
    assert np.unique(d).size == 1
    assert G.rms_granularity(d) < 1e-12


def test_grain_needs_a_pitch(grained_silver):
    """The seam rule, same as registration: a count lives on a sheet,
    and the sheet belongs to the caller."""
    neg = np.full((4, 4, 3), 0.5, np.float32)
    with pytest.raises(ValueError, match="pitch_um"):
        process_print(neg, 1.0, "silver", grain=True)


def test_an_unsourced_particle_refuses(monkeypatch):
    """Abstention is not invention: a process whose image particle no
    source has measured does not get a made-up one."""
    monkeypatch.delitem(PROCESSES["silver"], "grain_um2", raising=False)
    neg = np.full((4, 4, 3), 0.5, np.float32)
    with pytest.raises(ValueError, match="abstention"):
        process_print(neg, 1.0, "silver", grain=True, pitch_um=PITCH)


def _sigma_and_density(name, dose, shape=(256, 256), block=1, seed=0):
    img = G.uniform_print(name, dose, shape=shape,
                          grain=True, pitch_um=PITCH, seed=seed)
    d = G.density_field(name, img)
    return G.rms_granularity(d, block), float(d.mean())


def test_the_fluctuation_is_saunders_law(grained_silver):
    """sigma_D = sqrt(kappa*a*D/A), no dial in sight. Checked at
    three densities through the whole print-and-read chain."""
    for dose in (0.35, 0.8, 2.0):
        sigma, dmean = _sigma_and_density("silver", dose)
        want = float(np.sqrt(KAPPA * A_TEST * dmean / PITCH ** 2))
        assert sigma == pytest.approx(want, rel=0.03), \
            f"dose {dose}: sigma {sigma:.5f} against Saunders {want:.5f}"


def test_selwyn_invariance(grained_silver):
    """sigma*sqrt(2A) must not care what aperture reads it - the
    classical signature that noise is GRAIN, with no scale of its
    own, rather than an effect with a radius."""
    img = G.uniform_print("silver", 0.8, shape=(512, 512),
                          grain=True, pitch_um=PITCH, seed=3)
    d = G.density_field("silver", img)
    s = [G.selwyn_coefficient(d, PITCH, block=b) for b in (1, 2, 4, 8)]
    assert max(s) / min(s) < 1.08, s


def test_the_mean_is_preserved(grained_silver):
    """Grain redistributes the deposit; it must not expose the print.
    The count is drawn about the Nutting mean and converted back
    through the same relation, so this is exact up to sampling."""
    plain = G.density_field("silver", G.uniform_print("silver", 0.8))
    grained = G.density_field("silver", G.uniform_print(
        "silver", 0.8, grain=True, pitch_um=PITCH, seed=5))
    assert float(grained.mean()) == pytest.approx(
        float(plain.mean()), abs=8e-4)


def test_variance_grows_linearly_with_density(grained_silver):
    """The density dependence that makes grain live in the shadows'
    counts rather than in a uniform overlay: sigma^2 proportional to
    D, straight from sigma_N = sqrt(N)."""
    s1, d1 = _sigma_and_density("silver", 0.35, seed=11)
    s2, d2 = _sigma_and_density("silver", 2.0, seed=12)
    assert (s2 / s1) ** 2 == pytest.approx(d2 / d1, rel=0.1)


def test_one_population_of_particles_colours_every_channel(grained_vandyke):
    """A real print's grain is monochrome within a process: one
    realised deposit, coloured by the process's own absorption. The
    per-channel densities must be the SAME field scaled by absorb -
    pixel by pixel, not just on average."""
    img = np.asarray(G.uniform_print(
        "vandyke", 0.8, shape=(64, 64),
        grain=True, pitch_um=PITCH, seed=9), np.float64)
    sheet = np.asarray(process_print(
        np.zeros((1, 1, 3), np.float32), 1.0, "vandyke"), np.float64)[0, 0]
    dch = [-np.log10(img[..., c] / sheet[c]) for c in range(3)]
    absorb = PROCESSES["vandyke"]["absorb"]
    for c in (0, 1):
        assert np.allclose(dch[c] * absorb[2], dch[2] * absorb[c],
                           rtol=0, atol=2e-3), \
            "the channels carry different grain fields"


def test_the_count_is_seeded(grained_silver):
    """Deterministic per machine, like every develop stage: the same
    seed is the same print to the bit, and a different seed is a
    different sheet of the same paper."""
    kw = dict(grain=True, pitch_um=PITCH)
    a = G.uniform_print("silver", 0.8, shape=(32, 32), seed=7, **kw)
    b = G.uniform_print("silver", 0.8, shape=(32, 32), seed=7, **kw)
    c = G.uniform_print("silver", 0.8, shape=(32, 32), seed=8, **kw)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_every_particle_wears_its_source():
    """The synthesis table of the film-grain dossier, as constants:
    photolytic silver at Ware's 20 nm median (salt, vandyke), albumen
    at Reilly's MEASURED 8.1 nm historic-print mean, platinum at
    Ravines' 15-25 nm micrographs, cyanotype at Sentoku's measured
    within-fibre ceiling, silver at the conservation-TEM 0.5 um
    filament bundle, gum at its pigment class's geometric mean. No
    particle is invented; a process that loses its source loses its
    number and returns to the refusal."""
    want = {"salt": 3.14e-4, "vandyke": 3.14e-4, "albumen": 5.2e-5,
            "platinum": 3.14e-4, "cyanotype": 1.96e-3,
            "silver": 0.196, "gum": 8.0}
    for name, a in want.items():
        assert PROCESSES[name]["grain_um2"] == a, name


def test_smoothness_is_a_prediction_now():
    """The organ's point, in three orderings that were reputations
    and are now arithmetic. At the same pitch and dose: a platinotype
    prints far smoother than a silver-gelatin sheet (15-25 nm
    particles against 0.5 um bundles - the celebrated platinum
    smoothness, from a micrograph); gum prints visibly coarser than
    silver (micron pigment aggregates); and a salt print is
    grainless for any eye. None of these numbers was tuned to make
    this test pass - that is the entire claim."""
    pitch = 100.0                       # a contact-scale print pixel
    sig = {}
    for name in ("platinum", "silver", "gum", "salt"):
        img = G.uniform_print(name, 1.5, shape=(256, 256),
                              grain=True, pitch_um=pitch, seed=21)
        sig[name] = G.rms_granularity(G.density_field(name, img))
    assert sig["platinum"] < sig["silver"] / 10.0, sig
    assert sig["gum"] > sig["silver"] * 3.0, sig
    assert sig["salt"] < 3e-4, sig


FINE = 1.0          # um - a pitch near the crystal, the resolved regime


def test_the_sheet_exists_before_the_light():
    """The emulsion-first formulation's first claim: the crystal
    field is laid at coating, from the seed alone. A saturating
    exposure develops every crystal, so two DIFFERENT saturating
    exposures on the same seed print the identical field - the sheet
    itself - and a different seed prints a different sheet."""
    kw = dict(shape=(96, 96), grain=True, pitch_um=FINE)
    a = G.uniform_print("silver", 50.0, seed=4, **kw)
    b = G.uniform_print("silver", 80.0, seed=4, **kw)
    c = G.uniform_print("silver", 50.0, seed=5, **kw)
    assert np.array_equal(a, b), "the ceiling is not the sheet's own"
    assert not np.array_equal(a, c)


def test_more_light_never_undevelops_a_fixed_sheet(monkeypatch):
    """The coupling that makes it a sheet: every crystal carries a
    fixed sensitivity threshold laid at coating, so raising the
    exposure can only develop MORE of them, pixel by pixel.
    Independent draws per exposure - the organ's first formulation -
    invert this on roughly half the mass between neighbouring doses;
    no overlay of accurate patterns can pass it."""
    kw = dict(shape=(128, 128), grain=True, pitch_um=FINE)
    fields = [G.density_field("silver", G.uniform_print(
        "silver", dose, seed=11, **kw))
        for dose in (0.2, 0.6, 1.5, 6.0, 50.0)]
    for lo, hi in zip(fields, fields[1:]):
        assert np.all(hi - lo > -1e-9), \
            "a cell undeveloped as the light rose"


def test_a_pixel_smaller_than_the_crystal_refuses():
    """The honesty floor, per process: a cell narrower than the
    particle cannot hold an independent count - silver's bundles are
    0.5 um, gum's aggregates 3.2 um - and the emulsion refuses by
    name rather than printing a lie."""
    neg = np.full((4, 4, 3), 0.5, np.float32)
    with pytest.raises(ValueError, match="honest"):
        process_print(neg, 1.0, "silver", grain=True, pitch_um=0.3)
    with pytest.raises(ValueError, match="honest"):
        process_print(neg, 1.0, "gum", grain=True, pitch_um=2.0)
    # and exactly at the floor it prints
    process_print(neg, 1.0, "silver", grain=True, pitch_um=0.51)


def test_saunders_law_survives_in_the_resolved_regime():
    """The marginal must not have moved: near the floor, where the
    exact binomial-quantile path runs and lambda is a handful of
    crystals, the fluctuation is still sqrt(kappa*a*D/A) - the
    thinning theorem, now with the sheet as a real object."""
    img = G.uniform_print("silver", 0.8, shape=(256, 256),
                          grain=True, pitch_um=FINE, seed=6)
    d = G.density_field("silver", img)
    want = float(np.sqrt(
        KAPPA * PROCESSES["silver"]["grain_um2"] * d.mean() / FINE ** 2))
    assert G.rms_granularity(d) == pytest.approx(want, rel=0.03)


def test_the_increments_carry_the_thinning_law():
    """The reason crystals carry THRESHOLDS rather than each cell one
    luck value. Between two exposures of one sheet, thinned-Poisson
    physics makes the newly developed count its own Poisson:
    Var(n2 - n1) = lambda_K * (p2 - p1). WATCHED FAILING under the
    fixed-luck quantile coupling 2026-08-29: wanted 2.58, measured
    0.44 - six-fold under-dispersed increments, invisible to every
    single-exposure statistic and fatal to exposure ladders, double
    exposure, and any future Sabattier on the realised field."""
    lam = 2.1 * FINE ** 2 / (KAPPA * PROCESSES["silver"]["grain_um2"])
    kw = dict(shape=(256, 256), grain=True, pitch_um=FINE)
    step = KAPPA * PROCESSES["silver"]["grain_um2"] / FINE ** 2
    n1 = G.density_field("silver", G.uniform_print(
        "silver", 0.55, seed=9, **kw)) / step
    n2 = G.density_field("silver", G.uniform_print(
        "silver", 0.75, seed=9, **kw)) / step
    dp = float(n2.mean() - n1.mean()) / lam
    assert float(np.var(n2 - n1, ddof=1)) == pytest.approx(
        lam * dp, rel=0.1)


def test_a_stored_sheet_prints_the_same_print():
    """The film stock claim: coat once, develop against the stored
    crystals, and the print is bit-identical to the throwaway path
    with the same seed - a stock is a cache of the sheet, not a
    second model."""
    from atlas_film import emulsion
    pr = PROCESSES["silver"]
    frac = np.linspace(0.05, 0.95, 96 * 96).reshape(96, 96)
    K, thr = emulsion.coat(frac.size, 2.1, pr["grain_um2"], FINE, seed=13)
    stored = emulsion.develop_on(K, thr, frac.reshape(-1))
    fresh = emulsion.expose(frac, 2.1, pr["grain_um2"], FINE, seed=13,
                            label="silver")
    assert np.array_equal(stored, fresh.reshape(-1))


def test_grain_off_is_the_old_path_exactly():
    """grain=False must be the pre-organ arithmetic to the bit - the
    goldens' guarantee, asserted here as well as there."""
    rng = np.random.default_rng(2)
    neg = rng.uniform(0.0, 2.0, (16, 16, 3)).astype(np.float32)
    assert np.array_equal(process_print(neg, 1.0, "silver"),
                          process_print(neg, 1.0, "silver", grain=False))
