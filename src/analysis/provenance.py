"""Parameter-provenance table for the JTB manuscript (honest, ranges + source)."""

_ROWS = [
    {"name": "k_pers", "value": "0.001-0.05", "units": "/h",
     "source": "persister-formation range, literature-informed"},
    {"name": "mu_mut", "value": "1e-6", "units": "/cell/gen",
     "source": "heteroresistance/SCV emergence, illustrative"},
    {"name": "k_kill_base", "value": "1e-9-1e-7", "units": "/mL/h",
     "source": "immune killing capacity, illustrative"},
    {"name": "k_path", "value": "0.02-0.20", "units": "/h",
     "source": "pathogen-driven injury, illustrative (damage-response framework)"},
    {"name": "k_infl", "value": "0.005-0.10", "units": "/h",
     "source": "inflammation-driven injury, illustrative"},
    {"name": "k_heal", "value": "0.02-0.30", "units": "/h",
     "source": "host recovery, illustrative"},
    {"name": "I50", "value": "5.0", "units": "fold-change",
     "source": "inflammatory intensity at half-max injury, illustrative"},
    {"name": "alpha_cidal / alpha_static", "value": "3.0 / 1.0", "units": "-",
     "source": "relative lysis-driven cytokine release (TLR9), illustrative"},
]


def provenance_rows():
    return [dict(r) for r in _ROWS]


def provenance_markdown():
    header = "| Parameter | Value/range | Units | Source |\n|---|---|---|---|\n"
    body = "".join(f"| {r['name']} | {r['value']} | {r['units']} | {r['source']} |\n"
                   for r in _ROWS)
    return header + body
