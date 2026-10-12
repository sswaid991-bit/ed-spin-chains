# Practical Limits and Numerical Reliability in
# Exact Diagonalization of Quantum Spin Chains

Reproducibility package for the manuscript by
Saleh Swaid and M. Imran Jamil.

## Contents

- `paper.tex`      — LaTeX source of the manuscript
- `reproduce.py`   — Python script that regenerates every table, figure, and CSV
- `requirements.txt` — Python dependencies
- `figures/`       — publication figures (PDF)
- `results/`       — logged calibration sessions with raw CSVs

## Reproduce

    python -m venv venv
    venv/Scripts/activate        (Windows)
    source venv/bin/activate     (macOS / Linux)
    pip install -r requirements.txt
    python reproduce.py

Outputs go to `results/<session_id>/`.

## Logged sessions

- `results/20261007_070101/` — main session; all values in the paper
