# Biofilm Poroelastic Simulation Code

Companion code for the manuscript:

**Interfacial drainage regime as a coordinate linking biofilm mechanics and transport**

## Overview

Version **v3.1.0**. Three Python scripts regenerate every figure in the
manuscript (Figures 1-6 and Supplementary Figure S1) and write the data
plotted in each figure.

| Script | Produces |
|---|---|
| `biofilm_poroelastic_sim.py` | Figures 2-5 and S1, with one CSV of plotted data per figure |
| `figure1_drainage_boundary_access.py` | Figure 1 (PDF, TIFF and PNG) |
| `figure6_clinical_active_architecture.py` | Figure 6 and `Figure_6_region_bounds.csv` |

The main script solves one-dimensional linearised Biot/Terzaghi pressure
diffusion with hard and leaky interior sinks (Figures 2 and 3), evaluates
the analytical Mandel-Cryer solution (Figure 4), and runs a synthetic
model-recovery study (Figure 5). The header of each script describes the
model, the parameter values and the outputs.

## Running the code

    python -m pip install -r requirements.txt
    python biofilm_poroelastic_sim.py
    python figure1_drainage_boundary_access.py
    python figure6_clinical_active_architecture.py

No input files or options are needed. The main script takes about
4 minutes and 1.5 GB of memory. All outputs are written to
`biofilm_sim_outputs/`:

- `Figure_1.pdf`, `Figure_1.tiff` and `Figure_1.png` (600 dpi)
- `Figure_2` to `Figure_6` and `Figure_S1_Validation`, each as PNG and
  LZW-compressed TIFF, one file per figure with all panels assembled
  (500 dpi; 1000 dpi for Figure S1)
- CSV files with the data plotted in each figure

`validation_log.txt` holds the expected console output. It repeats the
numbers quoted in the manuscript, so a new run can be checked line by line.

## Requirements

Python 3.8 or newer with NumPy, SciPy, Matplotlib and Pillow; minimum
versions are listed in `requirements.txt`. Tested with Python 3.13,
NumPy 2.3.5, SciPy 1.17.1, Matplotlib 3.10.9 and Pillow 12.3.0.
Figures use Arial when it is installed and DejaVu Sans otherwise; the
font changes text widths, not numbers.

## Citation

If you use this software or the generated data, please cite the
accompanying manuscript:

> Patil, N. (2026). *Interfacial drainage regime as a coordinate linking biofilm mechanics and transport.*

The Concept DOI for all versions of this software is
[10.5281/zenodo.20813796](https://doi.org/10.5281/zenodo.20813796).
Zenodo assigns each release its own version DOI.

## License

MIT License. See [LICENSE](LICENSE) for the full license text.

## Contact

Navinkumar Patil
