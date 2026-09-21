# Bounded Partial Metric Segmentation

Code and numerical experiments accompanying the manuscript **“Aggregation of Bounded Partial Metric Spaces: The Product-Space Case”** by Arnau Mir-Fuentes and Oscar Valero.

The repository contains the liver-segmentation experiment used to illustrate the product-space aggregation framework developed in the manuscript.

## Compared methods

Three variants of the same modified fuzzy c-means procedure are considered:

1. **Classical method without aggregation**
   `D0(g,h) = |g-h|`.

2. **Classical product-space aggregation**
   `DC(u,v) = p_sp(u_sp,v_sp) + 7 |u_gray-v_gray|`.

3. **Partial-metric product-space aggregation**
   `DP(u,v) = p_sp(u_sp,v_sp) + 7 p_gray(u_gray,v_gray)`, where

   `p_gray(x,y) = |x-y| / (0.4 + max{x,y}) + 0.2`.

The spatial component is the Euclidean metric on normalized pixel coordinates.

## Experimental parameters

- Number of clusters: `k = 5`
- Fuzziness parameter: `m = 2`
- Maximum iterations: `100`
- Convergence tolerance: `1e-5`
- Random state: `42`
- Gray-level weight in the product-space variants: `7`
- Partial-metric parameters: `c = 0.2`, `t = 0.4`
- Numerical lower bound for dissimilarities: `1e-12`
- Body-mask threshold: `0.03`
- Erosion fraction: `0.05`
- Minimum erosion radius: `5`
- Morphological closing: `3 x 3`

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   └── run_experiment.py
├── data/
│   └── README.md
└── results/
    ├── paper_results.csv
    └── paper_summary.csv
```

## Data

The MRI/DICOM images and manually delineated reference masks are **not distributed in this repository**. The experiment in the manuscript is illustrative and the reference masks were not independently validated by a radiologist.

See `data/README.md` for the expected local directory structure.

## Installation

A recent Python 3 installation is required. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment and run:

```bash
pip install -r requirements.txt
```

## Running the experiment

After placing the local data in the structure described in `data/README.md`, run:

```bash
python scripts/run_experiment.py
```

The script executes the three clustering variants for the three images, applies the same cluster-selection and morphological post-processing procedure, evaluates the resulting masks, and writes the outputs to `results/`.

## Reported results

The numerical values reported in the manuscript are included in:

- `results/paper_results.csv`
- `results/paper_summary.csv`

The experiment uses only three MRI slices and is intended as an illustration of the mathematical framework, not as a clinical validation or as evidence of statistical superiority of one segmentation method.

## Manuscript

Arnau Mir-Fuentes and Oscar Valero, *Aggregation of Bounded Partial Metric Spaces: The Product-Space Case*. Manuscript.
