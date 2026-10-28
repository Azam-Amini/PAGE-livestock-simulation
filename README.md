# Stochastic Simulation of Gene Editing in Livestock Breeding Programs

This repository contains the complete simulation code, results, figures, and summary tables for the paper:
"Simulating the Benefits and Risks of Gene Editing in Livestock Breeding Programs"

## Contents

- `sim.py`: Main simulation (Python 3.10 + NumPy + Matplotlib)
- `make_figures.py`: Figure and table generation from the archived arrays
- `results/main.npz`: Result arrays for the nine main scenarios (50 replicates × 25 generations)
- `results/sens.npz`: Result arrays for the 12-configuration sensitivity analysis (24 replicates)
- `results/meta.json`, `results/summary.json`: Scenario metadata and run summary
- `figures/`: Manuscript figures, 300 dpi PNG and vector PDF (fig01–fig12)
- `tables/`: Recomputed summary tables:
  - `table3_recomputed.csv` (Table 3 of the manuscript, with `fix_frac` and `fix_prob`)
  - `tail_probabilities.csv` (Table 4)
  - `sensitivity.csv` (Table 5)

## Requirements
- Python 3.10+
- NumPy
- Matplotlib

## Reproduction
```bash
python sim.py
python make_figures.py
