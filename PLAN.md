# Implementation plan - what the medium grows

*Written 2026-08-28, at the repository's birth. The darkroom's
`docs/film-extraction.md` holds the extraction mechanics, the gate,
and the medium ledger; this file is the model half - what this
package must provide so those rows can close. The standing rules are
the siblings': nothing here may open a GL context or know the
darkroom exists; every referee is built and watched failing before
the organ it referees; numbers ship with sources or ship as declared
derivations; corrections that change prints land as named commits
with the golden rows re-baked and the why recorded.*

## 1. The reconciliation organ (first, because the evidence is in hand)

The optics-film-process dossier already measured the disagreements:
platinum dmax 1.30 against Ware's ~1.45; the salt/albumen scale
ordering inverted against Ware and Reilly; the shipped platinum curve
evaluating into Ware's palladium band; the pigment absorb 0.88
drifted from HUE_K = 0.92 against the constant's own comment
contract. The fix is a JOINT fit of dmax and toe per process against
Ware's own tables (Platinomicon p. 158 and p. 235, Cyanomicon
p. 276) - the dossier proved that raising platinum's dmax alone
worsens the exposure-scale overshoot from 2.107 to 2.164 - refereed
by a contrast-index and Dmax reader over the model's own curves, and
landed as breaking-latitude corrections with the film golden rows
re-baked, named. Vandyke/albumen/salt/gum dmax stay as wanted
sources; abstention is not invention.

*Landed 2026-08-28, two named commits. The referee
(`atlas_film/sensitometry.py`) reads Ware's own convention off
`process_print`'s actual prints, reproduced the dossier's independent
column to +/-0.004, and was watched failing before any constant
moved. The joint fit: platinum dmax 1.30 -> 1.45 with toe 0.85 ->
1.01 (reads Dmax 1.450 / scale 1.902 against Ware's measured ~1.45 /
~1.9), salt toe 1.25 -> 0.95 (prints its classmate's 1.90, above
albumen's 1.741 - the inversion undone). The absorb pair: HUE_K moved
to processes.py, both absorb lines read the name, 0.88 landed back on
the recorded 0.92. Golden rows re-baked darkroom-side with the whys
riding (`process:platinum`, `process:salt`, then the pigment-path
rows). Still open and wanted: the four unsourced dmax constants, and
gamma - the third measured number a single-exponent curve cannot hit
while holding Dmax and scale.*

## 2. Film grain

The ledger's oldest parked row. Emulsion grain is a counting
phenomenon in the DEPOSIT, distinct from the paper grain the darkroom
applies at print time: its statistics (Selwyn granularity, the
density-dependence of RMS grain) are the referee, built first from
the literature through the darkroom's sources program, then the organ
- a density-correlated noise field applied to the negative's own
measure, with the microdensitometer test as the gate.

*Landed 2026-08-28, down to the particle. The film-grain dossier
(darkroom `docs/sources/dossiers/film-grain.md`, 53 entries in four
lanes - the local Ware monographs plus three research sweeps) put a
measured or class-median particle under every process, and the organ
is Nutting-both-ways: `process_print(grain=True, pitch_um=..)` draws
each pixel's particle count Poisson about the coverage mean and
converts it back through the same relation, so the mean is exact,
sigma_D = sqrt(kappa*a*D/A) with no dial, and Selwyn's invariant
holds by construction (`atlas_film/granularity.py` is the
microdensitometer that checks all of it off actual prints). The
orderings are predictions now: platinum smooth because Ravines
measured 15-25 nm, gum visibly grainy because pigment aggregates are
microns, salt grainless because photolytic silver is nanoparticle.
Declared open: POP's size-with-count coupling (Reilly's TEM),
size distributions and the linear-vs-sqrt(D) tension, toning
geometry, tricolour-layer grain.*

*Deepened to emulsion-first 2026-08-29 (`atlas_film/emulsion.py`):
the sheet exists before the light - crystal field at coating
density from the seed alone, exposure thins it, development is the
deterministic quantile of fixed per-cell luck. Marginal law
unchanged to the digit (every referee test green untouched); gained:
sheet-before-light, pixelwise monotonicity in exposure, the ceiling
as the sheet's own realised field, and the honesty floor - grain
refuses pixels narrower than the process's crystal. The road this
opens, on the record: Sabattier and adjacency acting on the realised
crystal field rather than the mean.*

## 3. Reciprocity failure

The negative already records its shutter; long exposures on real
emulsions lose speed and gain contrast (Schwarzschild's p < 1). The
model is a per-process exponent with sources, the instrument is the
density ratio between a short and a long exposure of equal H, and the
gate is the published reciprocity curves of the processes that have
them.

## 4. Spectral sensitivity

The organ that closes the loop the lens side already opened: the
chemical-focus instrument measures where the actinic image forms, and
this organ says what each process DOES with each channel - cyanotype
and salt are blind past the blue, orthochromatic silver past the
green, panchromatic sees it all. A per-process channel-weight table
with sources, consumed at develop's dose line, making a red subject
print dark on an 1850s process because it must.

## 5. The migrations

Solarise, halation, paper grain and the tone shoulder are medium
physics living in the darkroom pipeline because they are coupled to
its blur machinery. Each moves here when its organ deepens it - the
extraction document's table is where that debt stays visible.

## 6. The camera stock: the negative becomes a sheet

*Planned 2026-08-31, before the work. This is the organ the project
has pointed at since the darkroom became the integrator: until now
the chain has been "a perfect plate behind a faithful lens,
contact-printed onto faithful paper" - every process in PROCESSES is
a PRINT medium, and the render reaches it as an ideal latent image.
This organ puts an emulsion at the film plane.*

**The chain.** The aerial image exposes a CAMERA STOCK - a real
named film with sourced constants - through the same emulsion
physics the print grain already earned: a crystal field coated from
a seed (storable, the film-stock machinery unchanged), thinned by
the stock's own response curve read as per-crystal probability,
developed all-or-nothing on fixed thresholds. The result is a
NEGATIVE: a density field with real grain at the negative's own
pitch, plus the stock's base+fog - which also counts crystals, so
the rebate and the shadows carry fog grain as real film does. The
paper process then prints THROUGH it: transmittance 10^-D becomes
the printing light, re-metered (the enlarger's exposure is its own
decision), optionally enlarged - same array, bigger sheet, so the
negative's grain magnifies onto the print exactly as an enlarger
magnifies it, and the paper's own grain rides on top at the print's
own pitch. Two sheets, two grains, compounding - which is what a
silver print of a film negative IS.

**Constants, sourced or refused.** `FILMS` carries per stock: dmax,
toe (fit jointly to the published contrast aim, the reconciliation
organ's discipline), speed, fog, grain_um2. The grain areas are
already in the dossier (Tri-X 17 -> 1.2 um2, Double-X 14 -> 0.82,
T-MAX 100's 8 -> 0.27, all via Saunders' inversion of published rms
granularity); the curve constants ship only when the sensitometry
lane lands, and a stock missing them refuses by name. Wet-plate and
dry-plate stocks are wanted sources, not inventions.

**What stays declared.** The enlarger is ideal (no enlarging-lens
PSF); spectral response is still the luma placeholder (organ 4's
home is now HERE, at the film stage, where panchromatic/ortho
belongs); reciprocity likewise (organ 3 - the datasheets' own
adjustment tables are its calibration and ride the dossier);
solarise remains a print-development model where it is; colour
negative film does not exist yet - a B&W negative prints every
process monochrome, which is a fact, not a gap.

**Referees before the organ:** the curve reader against published
contrast aims; the fog-grain claim (unexposed film fluctuates,
Poisson, about fog); the double inversion (bright scene -> dense
negative -> light print); monotonicity and the sheet laws inherited
through the chain; the enlargement bookkeeping (negative pitch vs
print pitch, both grains at their own scales). Golden rows for the
chain bake as new rows; no existing pin moves.

*Landed 2026-08-31, same day as planned, in four commits. The
mechanism first with the real stocks refusing their curves by name;
then the sensitometry lane (film-stocks dossier, 16 entries, the
read-off method validated by Double-X's printed gammas) and the
constants as its verdicts - with the curve family changed to
sensitometry's own on the way, because a camera negative holds
H&D's straight line for three decades and the print family cannot.
What was never fit emerged as the referees: trix's chord reads
0.564 against Kodak's printed CI aim of 0.56, and all four speed
points land within 0.04 log H of where ISO 6 puts their rated
speeds - the datasheet axes are absolute lux-seconds, so the model
meters cameras in real units by each stock's own curve. The chain
wired through develop(film=) with the double inversion, re-metering,
enlargement bookkeeping and no-sibling refusal all asserted, three
golden rows pinned, twenty untouched. Still open here: organs 3 and
4 now have their home at this stage, wet/dry plate stocks are
wanted sources, and the enlarger is ideal by declaration.*

## 7. Colour: the cinema chain

*Planned 2026-08-31, before the work, with the route chosen for
sourceability: VISION3 camera negatives printed onto 2383 print
film. The stills chain (Portra onto RA-4 paper) is deliberately
second - its modern sheets publish Print Grain Index, which Kodak's
own E-58 says cannot be compared with rms granularity, so its grain
has no honest route yet; the cinema sheets publish rms curves and
the print side has a real counterpart with published curves of its
own.*

**The model.** A colour negative is three of the sheets organ 6
built, stacked: blue-, green- and red-sensitive layers, each its
own crystal field with its own curve, seed and grain. Layer
exposures come from the render's channels through a per-stock 3x3
sensitivity matrix (derived from the published spectral-sensitivity
figures; the render is RGB, not a spectrum, and the matrix IS this
system's fidelity ceiling - declared). Each layer's developed dye
carries a per-dye absorption triple (read off the published
spectral-dye-density figures), and the three Status M channel
densities assemble as base + sum of dye contributions - the
integral orange mask is not modelled separately, because the
published per-channel D-min floors and curves already carry it.

**The assembly referee, before anything ships:** driven with a
NEUTRAL exposure sweep, the assembled model must reproduce the
datasheet's own three published curves within read-off tolerance.
The constants come from the same figures the referee checks against
- so the check is that the ASSEMBLY (matrix, curves, dyes, base)
is coherent, not circular curve-fitting; it is watched failing with
placeholder assembly first.

**Grain, honestly downgraded.** Dye clouds are not opaque discs -
Dutton's own derivation, already in the film-grain dossier,
excludes "partially transparent cells such as in a dye image". So
per-layer grain uses the same thinning machinery with a
GRANULARITY-EQUIVALENT area matched to the sheet's published rms
curves, declared as a noise constant and not a measured cloud size.
Colour grain here is honest about being one rung below the
crystal-literal claim the B&W sheets earned; the rung is named.

**Printing is LAD.** The print stage exposes 2383's three layers
through the negative's per-channel transmittance, and the printer
is calibrated the way laboratories calibrate: a mid-scale LAD grey
on the negative prints to the published aim densities on the print
film - per-channel printer exposures solved against sourced aims,
not metered by taste. Print-side grain by the same declared
granularity-equivalence; projection transfer declared.

**Wiring:** `film="5219"` (etc.) makes the negative colour;
`print_film="2383"` prints it; the B&W paper processes still accept
a colour negative through its transmitted luminance (a real
practice, declared as such). Golden rows bake as new rows; nothing
existing moves. Refusals everywhere constants are missing, by name.

*Landed (organ 7 constants, 2026-08-31): 50d and 5219 by layer
family fits (assembly vs traced curves: worst 0.046 / 0.036 D);
2383 by sampled layer tables - at gamma five the family missed the
printer-light criterion sixfold, and the traced geometry is the
source, so the layers carry it directly (worst 0.003 log H = 0.12
printer lights at all 27 traced points; construction declared in
the constants comment). LAD on both ends, grey-card trim beside
it, colour meter anchored on the G aim. One finding recorded: both
Vision3 stocks trace a B channel ~0.2 D thinner over G than
H-61A's nominal patch spacing - film greys print warm at pure LAD
lights; the B traces deserve a re-adjudication lane. Print-film
grain refuses by name (2383 publishes no rms curve; the FILM-F
lane's measured 1.25-4 um dye clouds in chromogenic PAPERS now
corroborate the negatives' granularity-equivalent areas from one
rung down).*

## Non-goals

Rendering anything (the darkroom's), print formats and sheet
geometry (the darkroom's - this package speaks microns and takes
pitch as a parameter), and any commons obligation: prints are studio
artifacts, and the negative's determinism story is the camera's.
