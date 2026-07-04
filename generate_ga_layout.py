import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.layout.baker import bake_layout

dsps = [(4, 1), (1, 1), (7, 2)]
brams = [(5, 2)]

width = 16
height = 16
aspect_ratio = 1.0

# 1. Bake Architecture XML
bake_layout(
    benchmark_name="custom_macbuf",
    dsps=dsps,
    mems=brams,
    width=width,
    height=height,
    output_path="visualizations/ga_custom_macbuf.xml",
    aspect_ratio=aspect_ratio,
)

# 2. Write Fake Place File
place_content = f"""Netlist_File: custom_macbuf.net Netlist_ID: SHA256:fake
Array size: {width} x {height} logic blocks

#block name\tx\ty\tsubblk\tlayer\tblock number
#----------\t--\t--\t------\t-----\t------------
"""

for i, (x, y) in enumerate(dsps):
    place_content += f"$mul~{i}[0]\t{x}\t{y}\t0\t0\t#{i}\n"

for i, (x, y) in enumerate(brams):
    place_content += f"bram_{i}\t{x}\t{y}\t0\t0\t#{i + len(dsps)}\n"

Path("visualizations/ga_custom_macbuf.place").write_text(place_content)
print("Created ga_custom_macbuf.place")
