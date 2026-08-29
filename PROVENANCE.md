# Provenance and credit

The code in this repository is MIT. The numbers it carries have their
own histories, and this file names them, because a process table whose
constants cannot be traced to their sources is a look wearing a
physics costume.

## Mike Ware — the process monographs

The historic process constants (dmax, toe, exposure scale) were built
by measurement-and-adjustment in the darkroom's own style, and then
held against Dr. Mike Ware's monographs — the *Cyanomicon* and the
platinum/palladium *Platinomicon*, freely published at mikeware.co.uk
— by the darkroom's external-sources program
(`atlas-darkroom/docs/sources/dossiers/optics-film-process.md`).

That audit found the shipped platinum dmax 1.30 contradicted by
Ware's own measured ~1.45 (three page-pinned statements), the
salt-versus-albumen tonal ordering inverted against both Ware and
Reilly, and the shipped "platinum" curve evaluating into Ware's
*palladium* exposure-range band. **The reconciliation organ closed
all three on 2026-08-28** as the joint fit the dossier demanded — a
one-number edit provably worsened the exposure-scale overshoot —
refereed by `atlas_film/sensitometry.py` (Ware's own convention, read
off this package's actual prints, watched failing first). Platinum
now prints his densitometry pair (Dmax 1.450, scale 1.902); salt
prints beside platinum-palladium, the class Ware put it in, above
albumen as both primaries order it. The derivations and page numbers
live where the numbers live (`atlas_film/processes.py`).

Vandyke, albumen, salt and gum dmax remain unsourced (Ware's
monographs are exhausted as a route to those four); they are wanted,
not hidden. So is gamma: Ware measured three numbers per sensitizer,
and a single-exponent curve holds two.

## James Reilly — the albumen and salted-paper literature

The salt/albumen ordering finding rests on Reilly alongside Ware; the
dossier carries the locators.

## This repository's own work

The saturating dose-to-deposit curve, the neutral-anchored tricolour
split (its derivation and its measured 6.8x-to-1.23x correction
recorded in the docstrings), and the subtractive paper law are this
project's constructions, corrected through the darkroom's audit
history that the carried commit log preserves.
