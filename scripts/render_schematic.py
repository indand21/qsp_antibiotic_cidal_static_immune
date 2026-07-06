"""
Render the Figure 1 model-architecture schematic from its Mermaid source.

Source of truth:  results/figures/manuscript/fig01_model_schematic.mmd
Outputs:          fig01_model_schematic.png  (raster, scale 3, white bg)
                  fig01_model_schematic.tiff (300 dpi, LZW, for submission)

Requires mermaid-cli (mmdc) on PATH and Pillow. Run from the project root:
    python scripts/render_schematic.py
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "results", "figures", "manuscript")
MMD = os.path.join(FIGDIR, "fig01_model_schematic.mmd")
PNG = os.path.join(FIGDIR, "fig01_model_schematic.png")
TIFF = os.path.join(FIGDIR, "fig01_model_schematic.tiff")


def main():
    if shutil.which("mmdc") is None:
        sys.exit("ERROR: mmdc (mermaid-cli) not found. Install: npm i -g @mermaid-js/mermaid-cli")
    res = subprocess.run(["mmdc", "-i", MMD, "-o", PNG, "-s", "3", "-b", "white"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("ERROR rendering PNG:\n" + res.stderr)
    print(f"wrote {PNG}")

    from PIL import Image
    im = Image.open(PNG).convert("RGB")
    im.save(TIFF, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    print(f"wrote {TIFF}  ({im.size[0]}x{im.size[1]}, 300 dpi)")


if __name__ == "__main__":
    main()
