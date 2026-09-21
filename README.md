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
├── CITATION.cff
├── LICENSE
├── environment.yml
├── requirements.txt
├── scripts/
│   └── run_experiment.py
├── data/
│   └── README.md
└── results/
    ├── README.md
    ├── paper_results.csv
    └── paper_summary.csv
```

## Data availability

The MRI/DICOM images and manually delineated reference masks used in the manuscript are **not distributed in this repository**. They must be available locally in order to rerun the experiment.

The reference masks are used only for evaluation; they are not used to fit the clustering models or to select the liver cluster. They were manually delineated for this illustrative experiment and were not independently validated by a radiologist.

See `data/README.md` for the expected local directory structure. Do not commit DICOM files or reference masks to this repository unless redistribution rights and de-identification have been verified.

## Installation

The repository has been tested with **Python 3.11**.

### Option A: Conda

From the repository root:

```bash
conda env create -f environment.yml --solver=classic
conda activate bounded-pm
```

If your Conda installation does not require the classic solver, `--solver=classic` may be omitted.

### Option B: venv + pip

Create an environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```text
.venv\Scripts\activate
```

or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the experiment

After placing the local data in the structure described in `data/README.md`, run from the repository root:

```bash
python scripts/run_experiment.py
```

The script executes the three clustering variants for the three images, applies the same cluster-selection and morphological post-processing procedure, evaluates the resulting masks, and writes generated outputs to `results/`.

The script currently has `SHOW_FIGURES = True`, so figures are displayed during execution. This can be set to `False` for a non-interactive run; it does not change the numerical procedure.

## Reported results

The numerical values reported in the manuscript are preserved separately from generated output:

- `results/paper_results.csv`: per-image values reported in the manuscript.
- `results/paper_summary.csv`: means reported in the manuscript.

See `results/README.md` for details.

The experiment uses only three MRI slices and is intended as an illustration of the mathematical framework, not as a clinical validation or as evidence of statistical superiority of one segmentation method.

## Reproducibility check

The repository was tested locally on Windows using a newly created Conda environment with Python 3.11. Running

```bash
python scripts/run_experiment.py
```

completed all nine experiments (three MRI images × three clustering variants) and reproduced the per-image FP, FN, misclassification rate, accuracy, Dice, and IoU values reported in the manuscript to the displayed precision.

For example:

- Image 1, classical aggregation: Dice = `0.8365`, IoU = `0.7189`.
- Image 2, partial aggregation: Dice = `0.7824`, IoU = `0.6425`.
- Image 3, no aggregation: Dice = `0.8710`, IoU = `0.7714`.

The current `scikit-image` version may emit a deprecation warning for `morphology.square`; this warning does not affect the numerical results of the verified run.

Because the image data are not distributed here, an independent rerun requires legitimate access to the same local input data and reference masks.

## Citation

If you use this code, please cite the accompanying manuscript and this repository. Citation metadata are provided in `CITATION.cff`.

## License

The **code and repository documentation** are released under the MIT License. See `LICENSE`. This license does not grant rights to any MRI data or reference masks, which are not included in the repository.

## Manuscript

Arnau Mir-Fuentes and Oscar Valero, *Aggregation of Bounded Partial Metric Spaces: The Product-Space Case*. Manuscript.
