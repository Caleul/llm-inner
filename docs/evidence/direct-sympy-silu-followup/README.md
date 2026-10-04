# Completing parts of two previously over-budget regions

Under the unchanged `4308c8e` numerical compiler identity, `run.py`
compiled same-sign rectangles whose broader parents had exceeded the
1 MiB flat-artifact budget. It admitted three fresh expressions covering
2,757,250 additional finite-Half input pairs after exact overlap removal.

There were 24,588 fresh actual-checkpoint/native comparisons, with zero
mismatches. Failed and unvisited subregions remain pending. The complete
coordinate was not emitted and multiple-token parity was not established.

`artifact-map.json` points to the effective input-only expressions;
`covers.json` retains source/checkpoint/backend identity, compilation and
parity reports. `frontier.json` captures the resulting geometry, including
both signed zeros and all unfinished inputs. Total covered cardinality
became 2,044,212,866 of 4,030,726,144 pairs.

This historical state is numerically incompatible with the later fixed-grid
conversion source. Its geometry was subsequently reset and freshly
recompiled as recorded in `../direct-sympy-fixed-grid/`. Do not reuse its
saved numerical expressions under that newer source identity.
