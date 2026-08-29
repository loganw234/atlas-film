"""The pigment-absorb calibration follows HUE_K, as its comment promised.

`HUE_K = 0.92` was named "because two places need the same number and
one of them is a calibration that has to follow if it moves" - and the
calibration did not follow. process_print's pigment-absorb line sat at
a literal 0.88 while the tricolour's sat at a literal 0.92 BESIDE the
named constant (the darkroom's findings queue #15; the film-process
dossier's round-1 out-of-scope note; inventory OPT-G7 records the site
as 0.92, so the code drifted after that survey rather than before it).

WATCHED FAILING 2026-08-28: every scan below found its literal before
the fix. The fix is structural rather than numeric - the constant
moved to processes.py, up the import direction the package already
has, both absorb lines read the NAME, and pigments re-exports it so
every consumer keeps its handle. A comment saying "has to follow" is
a hope; an import is a mechanism. The value lands at 0.92 because
that is the recorded number the drift departed from, which darkens
the pigment-path prints (gum, and any explicit ink) by design - the
darkroom's golden rows re-baked against the commit, old hashes
riding.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_film import pigments, processes

SRC_PROCESSES = (ROOT / "atlas_film" / "processes.py").read_text(
    encoding="utf-8")
SRC_PIGMENTS = (ROOT / "atlas_film" / "pigments.py").read_text(
    encoding="utf-8")


def absorb_line(src, marker):
    line = next(ln for ln in src.split("\n") if marker in ln)
    return line.split("#", 1)[0]


def test_the_process_side_reads_the_name():
    """The site that drifted. Its 0.88 was the finding."""
    code = absorb_line(SRC_PROCESSES, 'pigment or pr["pigment"]')
    assert "HUE_K" in code, "the pigment-absorb calibration is not the name"
    assert "0.88" not in code, "the drifted literal is back"


def test_the_tricolour_side_reads_the_name():
    """The site that held the right value - as a second literal, which
    is how the pair got two values in the first place."""
    code = absorb_line(SRC_PIGMENTS, "_hex(pig[i])")
    assert "HUE_K" in code
    assert "0.92" not in code, "the calibration is typed instead of derived"


def test_the_constant_is_defined_exactly_once():
    """pigments imports it; a second definition is the drift reborn."""
    assert "\nHUE_K = " in SRC_PROCESSES
    assert "\nHUE_K = " not in SRC_PIGMENTS
    imports = [ln for ln in SRC_PIGMENTS.split("\n")
               if ln.startswith("from atlas_film.processes import")]
    assert any("HUE_K" in ln for ln in imports), \
        "pigments no longer re-exports HUE_K"
    assert pigments.HUE_K == processes.HUE_K == 0.92


@pytest.mark.parametrize("module,print_once", [
    (processes, lambda: processes.process_print(
        np.full((2, 2, 3), 1.5, np.float32), 1.0, "gum")),
    (pigments, lambda: pigments.tricolour_print(
        np.full((2, 2, 3), 1.5, np.float32), 1.0, "carbro")),
])
def test_the_call_sites_read_the_name_at_run_time(module, print_once,
                                                  monkeypatch):
    """Not just the source: move the constant and the print must move,
    or the line captured a value once and the name is decoration."""
    before = print_once()
    monkeypatch.setattr(module, "HUE_K", 0.5)
    after = print_once()
    assert not np.array_equal(before, after), \
        f"{module.__name__} does not read HUE_K at its call site"
