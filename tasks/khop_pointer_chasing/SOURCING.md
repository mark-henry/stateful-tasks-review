# SOURCING — khop_pointer_chasing

Phase 1 (sourcing-only) per SPEC.md AMENDMENT. No `task.py` was written for this task.

Task recap: a SINGLE fixed random function/relation table f on N nodes, given once in the prompt;
state is the current node; step k applies f (the *same* f every step, unlike random_lookup_table
where each step gets a fresh table). Answer: node reached after `depth` hops. Important theory
note (per assignment): because f is fixed, repeated application admits a log-depth repeated-
squaring shortcut (precompute f, f^2, f^4, ... by doubling), so — despite the superficially serial
"chase the pointer" framing — this task is *not* provably forced-serial the way random_lookup_table
or s5_composition are.

## Candidates found and vendored

### 1. `vendor/Pointer_Value_Retrieval/` — cited paper Zhang, Raghu, Kleinberg, Bengio, "Pointer
Value Retrieval: A new benchmark for understanding the limits of neural network generalization"
(arXiv:2107.12580, 2021). Repo: https://github.com/qiuqiu1602/Pointer_Value_Retrieval, commit
`45e87ebfaa820ad0da0b76601c4fe61c48becdf1`. **No LICENSE file** (all-rights-reserved by default).

The repo's own README states plainly: *"This is my (unofficial) implementation and evaluation
on [the PVR paper]"* — a hobbyist reproduction, not an official release from the paper's authors.
(An earlier web-search summary mis-described this as "the official code release"; verified false
by reading the README directly — flagging this since it's the kind of error that would otherwise
silently propagate.) No official PVR code release from Zhang/Raghu/Kleinberg/Bengio was found.

**What it contains:** six Jupyter notebooks (`vec_pvr_*.ipynb`, `vis_pvr_*.ipynb`) reproducing the
*vectorized* and *visual* PVR tasks: a fixed-length input vector where a prefix "pointer" segment
encodes an index into a "value" segment, single-hop indirection (read pointer -> look up one
value), not an iterated/repeated k-hop chase. No generator module separate from notebook cells,
no license, no text/CoT prompt format, no `solve`/`check` split — everything is inline numpy in
notebooks aimed at CNN/MLP training, not autoregressive text generation.

**Fit to common interface:** none directly reusable — different modality (vectors/images, single
hop) and no license to redistribute derived code confidently.
**Verdict:** not usable as source; cite as the paper task family for the "indirection via a fixed
pointer structure" motivation only, and to justify why our task generalizes PVR's single hop to
k repeated hops over a fixed relation.

### 2. `vendor/Next-Token-Failures/` — Bachmann & Nagarajan, "The Pitfalls of Next-Token
Prediction" (arXiv:2403.06963, ICML 2024) — introduces the **path-star graph** task. Repo:
https://github.com/gregorbachmann/Next-Token-Failures, commit
`b925a1ef7c97f4beb533fd88a843b83932ea0df1`. **No LICENSE file found** (all-rights-reserved by
default; large pre-generated dataset dumps under `data/datasets/` — tens of MB of `.txt`/`.json`
files — were deleted from the vendored copy since only the generator code is of interest; commit
hash above still identifies the exact upstream snapshot the code came from).

**What it contains:** `data/graphs.py` (`star_graph(degSource, pathLen, numNodes)`): builds a
graph with `degSource` arms radiating from a source node, each arm a random path of length
`pathLen`, one arm leading to the true goal; the model is given the (shuffled) edge list and must
output the correct path from source to goal. This is genuine pointer-chasing-shaped graph
literature (chase edges to the goal) but structurally different from our task: (a) their relation
is a **sparse, non-total** edge list requiring *search over which arm to follow*, not a total
function f:[N]->[N] where the next hop is unambiguous; (b) the read-out is a **single next-token
prediction over the whole path**, no scratchpad/CoT; (c) their headline finding (the "Clever Hans
Cheat" under teacher forcing) is about a different failure mode (shortcut learning from
teacher-forced ground truth), not the repeated-squaring shortcut our task is designed to surface.

**Fit to common interface:** no reusable generator code (different combinatorial object — sparse
multi-arm graph search vs. dense total-function iteration) and no license; valuable purely as
adjacent literature for the README's "k-hop / pointer-chasing" citation list, per the assignment's
explicit mention of the k-hop compositional literature.
**Verdict:** not usable as source; cite as related work only.

### Also considered, not cloned
- HazyResearch/zoology (already vendored under `random_lookup_table/vendor/zoology`, MIT/Apache):
  its MQAR tasks are single-table parallel-query lookup, not iterated fixed-function hopping;
  same verdict as for random_lookup_table — cite, don't reuse.
- google-deepmind/neural_networks_chomsky_hierarchy (vendored under
  `s5_composition/vendor/neural_networks_chomsky_hierarchy`): no task there iterates a fixed
  random function; `cycle_navigation` iterates a *known, structured* (+1/-1/0) update, not a
  random lookup table, and is classification-only (no scratchpad).

## Recommendation

**Nothing is usable as-is, and nothing found is usable even with a thin wrapper.** Both leads
(PVR, path-star) are the closest thing this literature has to "pointer chasing," but neither
matches the specific mechanic required (repeated application of ONE fixed total random function
f:[N]->[N], read from a prompt-given table, tracked step-by-step in a text scratchpad). PVR is
single-hop and vector/image-based with no license; path-star is sparse-graph path *search* (multi-
armed, ambiguous-until-resolved) with no license and no CoT format. A `task.py` for this task must
be written from scratch. Both repos remain valuable purely as citations for the k-hop /
pointer-chasing motivation in the README and desk.json, and as documented reasons the
repeated-squaring shortcut caveat is worth flagging up front (visible already in the design, not
discovered post hoc).
