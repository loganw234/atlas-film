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

## Non-goals

Rendering anything (the darkroom's), print formats and sheet
geometry (the darkroom's - this package speaks microns and takes
pitch as a parameter), and any commons obligation: prints are studio
artifacts, and the negative's determinism story is the camera's.
