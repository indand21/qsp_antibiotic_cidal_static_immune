"""Render the model schematic (Figure 1; with the host-damage outcome).

Writes a Mermaid source that extends the model diagram with the D_host state
(injury from bacterial burden and from inflammation, with recovery) as the
model's outcome, then renders it with mermaid-cli (mmdc). The Mermaid text is
the source of truth and lives in this script; the .mmd/.png are regenerable
artefacts under results/ (gitignored).

Requires mmdc on PATH. Run from the project root:
    python scripts/phase3_schematic.py
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "results", "figures", "manuscript")
MMD = os.path.join(FIGDIR, "fig01_jtb_schematic.mmd")
PNG = os.path.join(FIGDIR, "fig01_schematic.png")

MERMAID = r"""%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 80, 'rankSpacing': 70}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart TB
    classDef pk   fill:#cfe2f3,stroke:#2c5aa0,stroke-width:1px,color:#111;
    classDef bact fill:#d5e8d4,stroke:#2e7d32,stroke-width:1px,color:#111;
    classDef imm  fill:#ffe6cc,stroke:#d79b00,stroke-width:1px,color:#111;
    classDef cyto fill:#f8cecc,stroke:#b85450,stroke-width:1px,color:#111;
    classDef dmg  fill:#e1d5e7,stroke:#8e44ad,stroke-width:2px,color:#111;
    classDef io   fill:#eeeeee,stroke:#777777,stroke-width:1px,color:#111;

    DOSE(["Drug dose"]):::io

    subgraph PK["Pharmacokinetics: two-compartment + effect site"]
        direction LR
        AC["A_central"]:::pk
        AP["A_peripheral"]:::pk
        AE["A_effect<br/>C_e (effect-site conc.)"]:::pk
        AC <-->|"Q"| AP
        AC -->|"K_p, k_e0"| AE
    end
    DOSE -->|"dose"| AC

    subgraph PD["Bacterial subpopulations"]
        direction LR
        BREP["B_rep<br/>replicating"]:::bact
        BPERS["B_pers<br/>persister"]:::bact
        BSCV["B_SCV<br/>small-colony variant"]:::bact
        BREP <-->|"k_pers"| BPERS
        BREP -->|"mutation<br/>(static pressure)"| BSCV
    end

    AE -->|"STATIC: Hill growth inhibition"| BREP
    AE -->|"CIDAL: direct + damage kill"| BREP

    subgraph HOST["Host response"]
        direction TB
        NEFF["N_eff<br/>immune effectors"]:::imm
        PAMP["PAMP pool<br/>(lysis-released)"]:::cyto
        IL6["IL-6"]:::cyto
        TNF["TNF"]:::cyto
        PAMP -->|"burst"| IL6
        IL6 -->|"ratio"| TNF
    end

    BREP -->|"burden recruits"| NEFF
    NEFF -->|"immune kill"| PD
    BREP -.->|"drives IL-6<br/>(cidal amplified)"| IL6
    AE -->|"cidal lysis<br/>releases PAMPs"| PAMP

    DHOST["D_host (host damage)<br/>OUTCOME: peak D_host"]:::dmg
    PD -->|"pathogen-driven injury<br/>k_path B/(B+B50)"| DHOST
    IL6 -->|"inflammation-driven injury"| DHOST
    TNF -->|"inflammation-driven injury"| DHOST
    DHOST -->|"recovery (k_heal)"| DHOST
"""


def main():
    # On Windows the launchable file is mmdc.cmd; prefer it so subprocess can find it.
    mmdc = shutil.which("mmdc.cmd") or shutil.which("mmdc")
    if mmdc is None:
        sys.exit("ERROR: mmdc (mermaid-cli) not found. Install: npm i -g @mermaid-js/mermaid-cli")
    os.makedirs(FIGDIR, exist_ok=True)
    with open(MMD, "w", encoding="utf-8") as f:
        f.write(MERMAID)
    res = subprocess.run([mmdc, "-i", MMD, "-o", PNG, "-s", "3", "-b", "white"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("ERROR rendering PNG:\n" + res.stderr)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
