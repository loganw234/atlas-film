# atlas-film

The medium of the Eidograph: what a film stock, a plate, a pigment
and a paper actually do to light, as numbers with provenance and the
response curves they parameterise.

Pure numpy. No GL, no plates, no darkroom import — the darkroom is the
instrument that *applies* these models, not a dependency of them. The
sibling repositories are [atlas-optical](../atlas-optical) (the glass)
and [atlas-darkroom](../atlas-darkroom) (the instrument that brings the
parts together).

Every constant here is sourced or refused by name. Where a datasheet is
silent the model raises with the reason and the dossier entry rather
than filling the gap with something plausible — because a number
invented for a named real film is a lie wearing a datasheet's name.

![Fourteen characteristic curves, density against log exposure, the
1890 plates three decades slower than modern
film](docs/curves.png)

*Every stock on the shelf, drawn straight out of `films.characteristic`
— the same expression a negative develops through. The flat runs at
each end are the sampled tables' own end clamps: past the last traced
point the model holds rather than extrapolates.*

## The shelf

**Camera stocks.** TRI-X 400, T-MAX 100, T-MAX 400, EASTMAN Double-X
5222, Plus-X 125, T-MAX P3200, FP4 Plus, HP5 Plus — each fit to its own
datasheet's read-off curve, on an axis ISO 6 proved absolute: every
stock's speed point lands within 0.04 log H of its rated speed,
unfitted.

**Colour.** VISION3 50D, 250D, 200T and 500T through to VISION Premier
2383 print film — three crystal-field layers a side, dye matrices from
the spectral-dye-density plates, LAD calibration at both ends.

**The era.** Two 1890 gelatin dry plates fit over Hurter & Driffield's
own printed tables, and wet collodion from the 1998 JIST recreation.

**Print processes.** Cyanotype, vandyke, salt, gum, platinum, albumen,
silver — the historic table, its saturating response curve, the
tricolour pigment model and the Beer–Lambert paper law.

## The organs

Ten so far, each one a named way a real negative and this one used to
differ under some instrument.

### Grain is a count, not a texture

![A 2.3 gigapixel print at 0.5 microns per pixel, individual grains
resolved as discrete steps](docs/grain.jpg)

The deposit is a Poisson count of *sourced* particles per pixel through
Nutting both ways, with no dial. The crystal field is laid at coating
time from the seed alone; exposure thins it; development is
deterministic amplification of fixed luck. Platinum smooth, gum grainy,
salt grainless are now predictions rather than settings. At 0.5 µm per
pixel — finer than one crystal — you can watch single grains step the
density by 0.34 D each.

### The plate's eye

![One negative read three ways: in colour, on a blue-blind 1890 plate,
and on panchromatic TRI-X](docs/spectral.jpg)

Every stock declares its spectral response. The Kodak four are flat
panchromatic; the 1890 plates are blue-only, "as black as Indian ink"
to the rest of the spectrum, on H&D's own tables. Put a subject that
encodes its structure in *hue* in front of a period plate and the warm
half of it simply does not record — Hardwich's vase, reproduced.

### The light that comes back

![The halation spread peaking at the critical radius, and the transfer
that shows it cannot double-count the MTF organ](docs/halation.png)

Light crosses the support, reflects at its back face, and returns to
re-expose at a radius the support's own thickness and index dictate.
The geometry is derived and then found to be old: Cornu read
ρ = 2e·tan R to the Académie des Sciences in **1890**, the same year as
the plates on this shelf. Only the 1890 plates halate here — they are
the one case where "no antihalation measure" is positively sourced
rather than merely unstated. The *strength* has never been published
for any photographic material, so it is the operator's number, named as
one, measured against a sourced ceiling.

![One lit point on an unbacked plate: the ring, with the predicted
radius drawn rather than fitted](docs/halation-ab.jpg)

### The emulsion's sharpness

![Twenty traced modulation-transfer curves, every black-and-white one
crossing 100 per cent](docs/mtf.png)

Twenty published MTF curves, traced and applied exactly in frequency
space at the negative's own pitch. The super-unity overshoot is kept
faithfully: chemical adjacency really does push TRI-X to 112% and
Double-X to 125%, and a model that could not exceed unity would deny
the most informative feature of the data. Stocks whose sheets publish
no MTF stay pixel-sharp, declared per stock.

### The cinema chain

![The subject, the orange negative with its mask, and the print timed
two ways](docs/colour.jpg)

A colour negative is three crystal-field layers with their own dye
matrices, an orange mask, and DIR coupling between the layers —
constructed exact-at-neutral so the traced curves, which are neutral
sweeps, stay bit-identical. The print is timed by LAD, with a grey-card
timing beside it, because the trim between the two is a real finding of
the traced constants rather than a preference.

### And the rest

Reciprocity failure per stock, with refusal outside the tabulated span
because silence is not zero. Development contrast on each sheet's own
CI-versus-time tables. The intensifier's bath. The batch lottery and
the hand-poured coating field, because the same boxed 1890 plate moved
actinograph speed 7 to 18 between purchases and the meter never knew.

## What it does not hold

Rendering, print formats and sheet geometry — those are the darkroom's;
this package speaks microns and takes pitch as a parameter. No lens: the
glass is atlas-optical's. And a long list of things it *refuses* rather
than guesses, each with a stated reason, from variable-contrast paper
machinery to a halation strength no laboratory ever published.

## Provenance

Born 2026-08-28 by extraction from the darkroom's develop monolith,
history carried, every print bit-identical across the move — the gate is
the darkroom's `tools/film_golden.py`, and the plan it followed is
`docs/film-extraction.md` there. The evidence behind every constant
lives in the darkroom's `docs/sources/dossiers/`, which is where the
fourteen-plus publisher defects caught along the way are recorded too.

`PLAN.md` here is the model half: what the medium still owes, and the
organs that will pay it.

## Checks

```
pip install -e .[test]
python -m pytest -q
```

Figures in `docs/` are generated by the darkroom's
`tools/readme_figures.py` (this package has no plotting dependency by
charter) and each carries a sidecar `.json` of the numbers it was drawn
from, so a figure can be checked against the model rather than believed.
