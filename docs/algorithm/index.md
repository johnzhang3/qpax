# Algorithm

`qpax` is a primal–dual interior-point (PDIP) solver. This section sketches
the shared structure and then walks through the *explicit* and *implicit*
variants of the three main routines:

- **[Core algorithm](core.md)** — the principles shared by both backends.
- **[Explicit algorithm](explicit.md)** — predictor–corrector PDIP from
  `cvxgen` (`backend="e"`), the default.
- **[Implicit algorithm](implicit.md)** — retraction-manifold PDIP
  (`backend="i"`), an alternative that updates slack/dual variables via a
  positive-cone retraction.
