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
250d and 200t the same day when the FILM-E lane returned (worst
0.022 / 0.029 D - the whole current Vision3 line is now on the
shelf, granularity-equivalent areas from the traced rms plates,
200t's red figure flagged as the dossier's weakest);
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

## Organ 4 - spectral response (camera side)

The film stage sees luminance through fixed luma weights - a
placeholder the extraction ledger has carried by name since organ
6. The organ retires it: each camera stock gains `sens=(r, g, b)`,
its response projected onto the render's three broad bands,
normalised to sum one so a neutral scene's metering is unchanged.

The projection is DECLARED, not fit: the render's channels carry
no spectra to integrate against, so a class projects to the
maximum-entropy weights consistent with its sheet - panchromatic
(all four Kodak stocks, FILM-N16: ~250-650 nm, cliff at 650-660)
projects flat (1/3, 1/3, 1/3); blue-sensitive plates (early-plates
D9/D13: blue/violet/UV only, "as black as Indian ink" to the rest)
project (0, 0, 1) with the UV lobe beyond the render's reach,
declared. The true peaks ride the comments (TRI-X 380-400 with a
560-620 shoulder; 5222 420-440). An ortho class exists the moment
a sourced ortho stock does. `negative`/`normal_exposure` accept
the full RGB aerial image and apply the STOCK's projection - the
projection is the film's property, not the darkroom's - while
scalar fields keep meaning already-projected light, so the law
tests read unchanged.

THE PAYOFF: the first two plate stocks. "manchester" (H&D's
Manchester Slow, early-plates D1: the complete 14-point 1890
curve, 0.625-5120 candle-metre-seconds) and "hd22" (D5: the
unnamed faster plate, 14 points, gamma 1.176 PRINTED by H&D
themselves - the fit's validator). One candle-metre-second is
declared ~ one lux-second on the model's absolute axis (D3, with
the candle's blue-poor caveat recorded); densities are net of fog
because H&D subtracted the fog strip, so fog=0 honours the
source's own convention (the silence recorded). Grain is
BRACKETED, wearing its flag: D19 is a recorded silence, and the
declared 0.196 um2 is the developed-silver particle class the
silver print already carries - inside Vitale's modern gelatin
range and consistent with collodion's microfilm-low granularity -
a bracket, not a measurement, named as such at the constant.
A NEW REFEREE reads the plates by H&D's own printed formulas
(D2): gamma = (D2-D1)/(logE2-logE1) and log-inertia by the
straight-line intercept, off the MODEL's curve - hd22 must give
back the printed 1.176, manchester the computed 0.89-0.91 band
and S = 34/i ~ 5.6 (D4). Collodion refuses by name: lane D holds
its exposure anchors and its microstructure but no traced curve,
and a curve invented for a named process is the lie this package
exists to refuse.

## Organ 3 - reciprocity

`negative`/`normal_exposure` gain `t=None`, the exposure duration
in seconds. None keeps today's reciprocity-free behaviour,
declared. With t, the stock's own sheet table (FILM-N13/N14)
compensates the exposure: H_eff = H * 2^(-comp(t)), comp
interpolated in log t between the sheet's own rows - TRI-X's
+1/2/3 stops at 1/10/100 s against T-MAX 400's nothing-to-1s, the
4x spread the dossier called out as per-stock, never shared.
Outside a sheet's tabulated span the stock REFUSES BY NAME (5222
is silent beyond 1 s, and silence is not zero). The plates carry
comp = 0 on H&D's own check (D3: 1/4 candle-metre for 40 s gave
the density of 1 for 10 s - reciprocity held; their whole axis is
I*t). The modelled negative is the COMPENSATED one: the sheets
pair their +stops with development cuts precisely to hold the
curve's contrast, so applying the exposure term against the
normal-development curve reproduces the properly-worked negative,
and the uncompensated contrast rise (which the dossier warns
about but no sheet quantifies) stays out rather than invented.
Colour stocks refuse t: no colour lane has delivered a
reciprocity table yet.

Wiring: develop() gains shutter= (seconds) feeding the film
stage; goldens re-pin neg:* (the luma placeholder retiring is a
model correction, the why rides the record) and bake plate and
shutter rows as new rows.

*Landed 2026-08-31, same day (atlas-film "the sheet gains an eye
and a clock"; darkroom wiring beside it): hd22's chord read back
1.165 against H&D's printed 1.176; manchester in the 0.89-0.91
band, S = 34/i 5.9 against the plate's 5.6; red-vs-grey splits
0.58 D on a plate and 0.00000 on TRI-X; meter and film cancel
bit-identically and the golden row pins the cancellation. 88
tests. Record: darkroom
docs/test-records/2026-08-31-the-eye-and-the-clock.md.*

## Organ 5 - the developer's hand, and the wet plate

Development time as a contrast dial, from the sheets' own traced
CI curves (developer-hand dossier, lane I: sixteen developer/time
combinations traced vector-exact from the PDFs' path operators,
residuals 0.0002-0.007 in data units). The lane's load-bearing
derivation: every Kodak-recommended 20 C time lands on contrast
index 0.553-0.571 - "normal development" IS CI ~ 0.56 - so
push/pull becomes a displacement along the stock's own curve.

The dial is `ci=`: the contrast the negative was developed to, in
the stock's own measure (contrast index diffuse-visual for the
Kodak four; gamma Status M for 5222, whose sheet prints five
gammas that fit a LINEAR time law to 0.007; H&D's development
factor for the plates, whose law - time scales gamma, ratios
fixed - the early-plates lane already holds). Implementation is
gamma scaling against the stock's normal contrast, span fixed:
the properly compensated negative, exactly as organ 3 argued.
Bounds are each table's own traced span, refusing outside; colour
refuses entirely (no lane); collodion refuses BY MECHANISM -
development time is not a contrast control there (45-90 s
"equally satisfactory", lane I): halide loading and
intensification are its knobs, and the intensification dial waits
for its own landing. The reciprocity tables' paired development
cuts stay declared as organ 3 left them. Corrections the lane
forced, recorded: T-MAX 100's normal is 8 3/4 min (not the 8 the
constants comment carried), T-MAX 400's is 7 1/2; TRI-X's pin (CI
0.56 at 6 min) is CONFIRMED at 0.554.

AND COLLODION TAKES THE SHELF. The 1998 JIST recreation delivered
what lane D could not: a traced characteristic curve (1.1%
iodide, iron development, 19 points off the one separable curve,
axis calibration 0.001/0.009 residuals), carried as a sampled
table because the bent shoulder is the source's own geometry
("densities below 1 are formed only by surface silver"). fog=0
on the net-density axis; blue-eyed (spectral max ~420 nm); grain
DECLARED granularity-equivalent at 0.196 um2 - the measured
microfilm-low granularity outranks the 4-6 um iodide particle
size because Dutton's opaque-disc assumption fails for
surface-stacked silver, and the conflict is recorded at the
constant. The absolute axis is BRACKETED and flagged twice over:
the authors' working index DIN -9 (~ISO 0.1, the lane's own
conversion, flagged) agrees with Towler's 1864 field exposures
within a stop. Reciprocity refuses (nothing sourced); ci refuses
(the mechanism, above); intensification is the named future dial.

## Organ 5b - the intensifier's bath

Collodion's real contrast dial (developer-hand I17): metol-silver
physical intensification, the operation D16 established as
ROUTINE practice, not remedy. The model: intensification deposits
silver on the developed image, so the output density field -
grain and all - multiplies by a factor; crystal statistics are
untouched because the crystals were already developed when the
bath touched them. The factor ships for the PICTORIAL recipe
only: intensify="1:10" multiplies by 2.6/1.57 = 1.656, the
sourced Dmax ratio, and the REFEREE is that the same factor must
reproduce the printed gradient endpoint (0.85 x 1.656 = 1.41
against the printed 1.37 - within 3%, two printed numbers, one
factor). The 1:5 line-work regime REFUSES BY SHAPE: the source
reports it straightens the curve ("a satisfactory, long linear
relationship without a sharp shoulder") and a pure scale cannot
honestly represent a shape change - it waits for a traced
intensified curve. Non-collodion stocks refuse by name (the dry
plates were mercury-intensified in period, Q 1.9 in Mees's table,
but no lane has traced what mercury does to THEIR curves).
"Intensification results in higher granularity" (I17) becomes an
EMERGENT PREDICTION rather than a dial: multiplying the developed
density field multiplies its grain sigma by the same factor.

## The enlarger's lens

The last ideal stage of the enlarger, closed with the same
declared physics the camera already carries: enlarger_fstop= and
enlarger_diffraction= apply the Airy blur to the PRINTING LIGHT
at the print's own pitch (the negative's pitch times the
enlargement), exactly as the camera dials apply it to the aerial
image at the negative's pitch. Off by default and off is not a
lie - an ideal enlarger lens is an honest instrument, the same
argument develop() has always made for the camera. No new
constants: diffraction is the optics dossiers' physics, reused at
the second lens position. Aberrated/traced enlarger lenses ride
atlas-optical's existing machinery when wanted; the darkroom
stage only owns the aperture.

*Landed 2026-08-31, the closing sweep: organ 5b's 1:10 bath
(factor 1.656, two printed endpoints, grain amplification
emergent) with the enlarger's lens beside it in the darkroom
(working aperture N(1+m), the f/128 golden lesson); the colour
clocks (every VISION3 sheet's flat 1/1000-1 s span refusing
beyond, the print stock's own 1/3000-1/10 s domain, push declined
with Kodak's own under-a-third-stop correction banked); and FOUR
MORE SHEETS as traced tables - Plus-X emerging on the ISO axis at
+0.017 while P3200's artwork sits 2.3 stops off its own EI
(re-anchored, labelled), the Ilfords anchored by explicit offsets
with the contrast criterion as validation. Eleven B&W stocks,
four colour negatives, one print stock, three plates-era media.
117 tests; 40 golden prints.*

## The structural campaign - sharpness, glare, and the layers talking

Three lanes dispatched 2026-08-31 against the three structural
gaps, landing in order of weight:

**Organ 8 - the emulsion's sharpness (MTF).** The sheets publish
modulation-transfer curves for every Kodak stock on the shelf and
the lane traces them (vector where the art allows, calibrated,
residuals stated - the developer-hand method). The organ:
`negative` convolves the aerial image with a kernel whose MTF
matches the stock's traced curve at the negative's pitch - the
emulsion's turbidity made spatial, the same declared-physics
pattern as the enlarger's Airy blur. Adjacency lift (MTF above
100% at low frequency) is real physics the trace must keep, and
the kernel family must be able to hold it (a difference-of-
gaussians, not a single blur). Stocks whose sheets print no MTF
refuse by name. Default ON once landed? No - sharpness belongs to
honesty: on by default at the film stage, with a kill switch,
because a pixel-perfect emulsion is the lie now.

**The era-look organs.** Landing NOW from sources already in hand
(early-plates D10): the plates' BATCH LOTTERY - the same boxed
product moved actinograph speed 7 to 18 between purchases, so
`batch=` (an integer, the box you bought) draws a deterministic
speed shift within the sourced order-of-magnitude bracket of
+/-1 stop, and the METER DOES NOT KNOW - the photographer rated
the plate at its nominal speed and the batch betrayed them,
which is the era experience the dial exists to reproduce. The
same hand poured the coating: D10's p. 198 strip varied D 1.335
to 0.820 across one supposedly uniform plate, so the same batch
seed lays a smooth low-frequency POURING FIELD that multiplies
the developed density (and its grain with it, like the
intensifier's bath) at the sourced worst-case amplitude. Plates
only: the Kodak stocks' machine coating has no sourced variance,
collodion's pour is not in the record, and both refuse by name.
VEILING FLARE waits on lane M's numbers: the organ will be a
sourced flare fraction added to the aerial image before the
film stage - uncoated-era glass lifting every shadow - with the
era classes the lane returns.

**Organ 9 - interimage (the layers talk).** Lane O hunts the
quantitative form: DIR-coupler cross-layer development coupling,
ideally a printed gamma matrix or single-channel-exposure curves
for a real stock. If the magnitude cannot be sourced, the organ
lands as a named refusal with the mechanism banked - the current
independent-layer model is the declared simplification either
way, and the referee that asserts layer independence gets a
comment naming what real film deliberately violates.

*Landed 2026-08-31, same day: lane O delivered the printed
conventional-negative gamma-ratio triple (R 1.49 / G 1.64 /
B 1.50) and the founding patent's driver law, and organ 9 shipped
exact-at-neutral (neutral sweeps bit-identical, asserted), the
ratio read back through the public surface on every stock. The
batch lottery and pouring field landed the same day from D10;
the flare organ landed when lane M returned with the full era
ladder (uncoated 0.6-6.3% by surface count, coated 1.4%,
multicoated 0.4%, the scene-level factor emerging per scene as
Jones & Condit measured). Of the three structural gaps, two are
closed and the third - the emulsion's sharpness - waits only on
lane L's traces.*

## Organ 10 - halation, the light that comes back

*Planned 2026-08-31, before the work. Lanes FILM-N (the physics
and geometry), FILM-Q (per-stock base and antihalation), FILM-R
(the plates, the period record, backed versus unbacked) dispatched
at the same hour.*

**The absence.** Halation exists in this system already and is in
the WRONG PLACE, which is worse than missing. `darkroom.halation`
was written when the render went straight to paper: it applies an
annulus to the AERIAL image, before any emulsion, with a hand-set
`halation_radius=140.0` and a hand-set strength, off by default,
and it does not know which sheet is loaded. `atlas_film` does not
contain the word. So the 1890 plates - whose defining signature is
a bright window blooming into a halo millimetres across, because
nobody put anything behind the glass - halate exactly as much as
rem-jet-backed VISION3, namely not at all.

**The mechanism, and why it belongs to the sheet.** Light that has
already exposed the emulsion scatters within it, crosses the
support, and meets the support/air boundary at the back. Beyond
the critical angle it is totally internally reflected; below it,
Fresnel returns a few percent. What comes back up re-exposes the
emulsion at a radial displacement set by the support's own
thickness and index. That is a statement about what a stock does
to light, so the constants are the medium's and the organ lands
in `atlas_film/halation.py` beside `mtf.py`, applied at the
negative's own pitch by the same discipline.

**The geometry is derived, not dialled.** With the emulsion
scattering into the base as a Lambertian source, the flux into
angle theta is sin(2 theta) d theta, the return lands at radius
r = 2 t tan(theta), and the surface density works out to

    p(r) = R(theta) cos^4(theta) / (4 pi t^2),  theta = atan(r / 2t)

- the cos^4 law again, arriving from a different direction. R is
Fresnel: a few percent below the critical angle, unity above. So
the point spread is a weak disc that CLIMBS to a peak at
r_c = 2 t tan(asin(1/n)) and decays outward: the bright-edged
annulus the period literature describes, with no free shape
parameter at all.

*(Correction, same day, before any constant landed: this
paragraph first said the spread JUMPS at r_c, and the arithmetic
says otherwise. Fresnel reflectance climbs to unity CONTINUOUSLY
as theta approaches the critical angle, so there is no
discontinuity - the profile rises to 7.9x the centre value with
its maximum exactly at r_c, then decays. The ring is real and the
plan's geometry is unchanged; the word was wrong.

And then the correction's own word was wrong in the other
direction, caught by its referee an hour later: "smoothly" is not
right either. 1 - R goes as sqrt(theta_c - theta), so the profile
reaches its peak with an INFINITE derivative - a CUSP, continuous
but not differentiable, the outside/inside ratio falling by
exactly sqrt(10) per decade of epsilon on its way to 1. Which is
the better answer to what the period sources were describing all
along when they called it a sharply defined edge. Recorded in
both directions because a plan that only records the corrections
that flatter it is not a record.
Two further facts the same check turned up, both load-bearing:
only 6.9% of the returned light falls INSIDE r_c and 60% inside
2 r_c, so the halo is far broader than the ring radius alone
suggests and the kernel must reach ~10 r_c to hold 98%; and the
planar integral of p closes on the angular one to the truncated
tail (0.5797 against 0.5840 at n=1.48), which is the derivation
checking itself.)*

Only
two numbers per stock are wanted - the support thickness and its
index - and both are datasheet facts. The kernel goes to frequency
space numerically, exactly as organ 8's transfer does; no closed
form is needed and none is faked.

**What must be sourced or refused.** The STRENGTH: what fraction
of the exposing light enters the base as wide-angle scatter, and
what the antihalation measure removes from it. An undercoat
absorbs on the way down and again on the way back, so a density D
attenuates by 10^(-2D) - a double pass the sources must confirm.
Rem-jet is near-total suppression. An unbacked plate removes
nothing, and the arithmetic above says roughly half of what
scatters downward comes back, which is why the effect was worth a
century of chemistry. Every stock whose sheet is silent on its
base REFUSES to halate rather than guess, by name, as usual.

**The honesty question the organ must answer, not dodge.** A
published MTF is measured on the real film INCLUDING whatever
halation it has, so an organ that applies both risks counting the
same light twice. The defence is that MTF curves are normalised
to unity at low frequency and a halo hundreds of microns wide IS
a low-frequency spread, so the normalisation divides most of it
out - but the ring's characteristic frequency (~2 c/mm) sits at
the very bottom of the traced range, which is uncomfortably close.
The traced curves' own low-frequency behaviour is evidence either
way and will be read before the constants land. Whatever it says
gets written down, including if it says the two organs overlap.

*(Answered the same day, by the geometry rather than by
assertion. The halation transfer is K(f) = the Hankel transform
of the unit-normalised spread, and it collapses fast: for a
132 um acetate base K falls to +0.06 by 1 c/mm, -0.11 at 2, and
is within 0.001 of zero from 5 c/mm upward; for a 1.2 mm plate it
is done by 0.5 c/mm. Every traced MTF on the shelf carries its
lowest datum at or above 2.5 c/mm. So across the ENTIRE measured
range halation contributes no SHAPE - only the flat factor
(1 - g) - and a curve normalised to unity divides exactly that
factor out. The two organs are separable in frequency: organ 8
owns 2.5 c/mm upward, organ 10 owns everything below, and the
normalisation is the seam. They do not double-count, and the
reason is arithmetic that can be re-run rather than a judgement
call. What follows from it, and is less comfortable: the
published MTF therefore carries NO information about g, so the
strength cannot be bounded from the curves and must come from
the lanes or be refused.)*

**Default and switch.** ON where the constants exist, by organ 8's
argument: a plate that cannot halate is the lie now. `halation=
False` kills it. A pitch too coarse to resolve r_c is a
bit-identical no-op (`out is img`), so only rows whose ring
resolves at the golden raster move, each re-pinned with its why.
The darkroom's old aerial dial stays exactly as it is - existing
work may lean on it - documented as the aesthetic sibling of
`bloom`, with the note that setting both counts the light twice
and that is the operator's choice to make.

**Referees, watched failing first.** A single lit pixel on an
unbacked plate, developed, its radial density profile read: the
inner edge of the ring must land at 2 t tan(theta_c) within a
stated tolerance - the geometry read back off an actual print,
the house method. The same test on a sourced undercoat shows the
sourced attenuation. Halation ADDS light, so the frame's total
dose rises by the reflected fraction and never falls. Coarse
pitch returns the input object identically. A stock with no
sourced support refuses by name.

## Non-goals

Rendering anything (the darkroom's), print formats and sheet
geometry (the darkroom's - this package speaks microns and takes
pitch as a parameter), and any commons obligation: prints are studio
artifacts, and the negative's determinism story is the camera's.
