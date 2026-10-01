# License and provenance

Our contributions use the MIT license. The root [LICENSE](https://github.com/hetland/cgrid-tools/blob/main/LICENSE)
also retains the terms for upstream material; distribute that complete file.
The package metadata records these combined terms rather than claiming that
MIT replaces the upstream licenses.

| Source | Relationship to this package | Terms retained |
| --- | --- | --- |
| [gridgen-c](https://github.com/sakov/gridgen-c/tree/222cc71f859d6f483a7ba7a435e5cb9f41977906) | CRDT algorithm and implementation lineage; copied reference fixtures | Sakov/CSIRO redistribution and non-endorsement notice |
| [octant](https://github.com/hetland/octant/tree/e25118c31537d8f7411a0040e5806796419ee3e2) | C-grid metric API and formulas | BSD three-clause terms |
| [pygridgen](https://github.com/hetland/pygridgen/tree/18d78c7737a4ee1722ec73238ea689a6527d025f) | Focus transforms and boundary API | Rob Hetland's 2012 MIT notice |
| [SCPACK](https://www.netlib.org/conformal/scpdbl) | Schwarz-Christoffel routines through gridgen-c | Permission and publication citation notice |

These links identify the upstream snapshots consulted during the provenance
audit, not a reconstructed commit history for every adapted line. The
repository has no recorded derivation history establishing a clean-room
implementation, so the notices conservatively preserve the upstream lineage.

## Triangle

Upstream gridgen-c bundles Triangle, whose distribution has additional
restrictions. This package contains no Triangle source and has no Triangle
runtime dependency: constrained triangulation starts from
[SciPy's Qhull-backed Delaunay triangulation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html).
The gridgen-c license's Triangle note is preserved verbatim in LICENSE to
retain the complete upstream notice; it describes that upstream distribution.

## Citation

For publications using the Schwarz-Christoffel routines, retain SCPACK's
requested reference: L. N. Trefethen, *Numerical computation of the
Schwarz-Christoffel transformation*, SIAM Journal on Scientific and Statistical
Computing **1** (1980), 82-102.
