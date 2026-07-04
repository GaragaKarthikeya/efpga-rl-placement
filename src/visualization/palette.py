"""
Shared figure palette for the paper — one warm, editorial scheme used across
every figure (Fig 1 scale, Fig 2 seed variance, Fig 3 layout). Muted enough to
read well in print, warm enough to look good on a pitch deck.

Convention:
  TERRA  = the hero accent — RL / zero-shot / optimized / DSP
  TEAL   = the second series — in-pool / BRAM
  SAGE   = logic fabric (CLB)
  SAND   = I/O and highlights
  GRAY   = neutral / in-size controls
"""

# text + surfaces
INK     = "#3a352d"   # primary text (warm charcoal)
SUBINK  = "#6f6757"   # secondary text
CREAM   = "#f4efe6"   # pale surface / band fills
CHANNEL = "#ddd4c2"   # routing channels between tiles
ZERO    = "#b8ad99"   # zero / reference rule

# core accents
TERRA   = "#c0603a"   # primary accent
TEAL    = "#3f6f74"   # secondary accent
SAGE    = "#93a98d"   # CLB / logic fabric
SAND    = "#cba14e"   # I/O / highlight
GRAY    = "#a89f8d"   # neutral / controls
RECLAIM = "#d8c19a"   # reclaimed area

# edges (darker shades for crisp tile borders)
TERRA_E = "#8f3f22"
TEAL_E  = "#2c4f53"
SAGE_E  = "#73886b"
SAND_E  = "#9a7528"

# layout fabric tints — each is the "unused" (pale) member of its resource family
TERRA_PALE = "#e3a982"   # DSP column, unoccupied — light terracotta
TEAL_PALE  = "#9cc2bd"   # BRAM column, unoccupied — light teal
CLB_PALE   = "#c7d2bf"   # CLB site, unoccupied — pale sage (same family as CLB)
IO_PALE    = "#ecd9bb"
