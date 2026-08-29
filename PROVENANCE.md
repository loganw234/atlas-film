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

That audit is carried here honestly rather than silently: **the
shipped platinum dmax of 1.30 is contradicted by Ware's own measured
~1.45** (three page-pinned statements), the salt-versus-albumen tonal
ordering is inverted against both Ware and Reilly, and the shipped
"platinum" curve evaluates into Ware's *palladium* exposure-range
band. These are the reconciliation organ's to fix — as a joint fit of
dmax and toe against Ware's tables, because the dossier proved a
one-number edit worsens the exposure-scale overshoot — and until that
organ lands, the constants ship exactly as the darkroom shipped them,
with the findings stated where the numbers live
(`atlas_film/processes.py`).

Vandyke, albumen, salt and gum dmax remain unsourced (Ware's
monographs are exhausted as a route to those four); they are wanted,
not hidden.

## James Reilly — the albumen and salted-paper literature

The salt/albumen ordering finding rests on Reilly alongside Ware; the
dossier carries the locators.

## This repository's own work

The saturating dose-to-deposit curve, the neutral-anchored tricolour
split (its derivation and its measured 6.8x-to-1.23x correction
recorded in the docstrings), and the subtractive paper law are this
project's constructions, corrected through the darkroom's audit
history that the carried commit log preserves.
