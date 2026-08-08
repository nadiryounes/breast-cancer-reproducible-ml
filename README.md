# Reproducible and Explainable Machine Learning for Breast Cancer Classification

This repository accompanies the manuscript:

**“Reproducible and Explainable Machine Learning for Breast Cancer Classification”**

It contains the version-pinned analysis code, environment specifications, canonical machine-readable results, and the figures used in the manuscript. The repository is intended to allow reviewers and readers to reproduce the reported computational analyses from a clean Python environment.

## Authors

- Younes NADIR — M2S2I Laboratory, ENSET Mohammedia, Hassan II University of Casablanca; National Higher School of Art and Design (ENSAD), Hassan II University of Casablanca — corresponding author
- Mohamed RACHDI — M2S2I Laboratory, ENSET Mohammedia, Hassan II University of Casablanca; National Higher School of Art and Design (ENSAD), Hassan II University of Casablanca
- Abdellah BAKHOUYI — M2S2I Laboratory, ENSET Mohammedia, Hassan II University of Casablanca; National Higher School of Art and Design (ENSAD), Hassan II University of Casablanca
- Lahcen AMHAIMAR — M2S2I Laboratory, ENSET Mohammedia, Hassan II University of Casablanca; National Higher School of Art and Design (ENSAD), Hassan II University of Casablanca
- Abderrahim KHALIDI — M2S2I Laboratory, ENSET Mohammedia, Hassan II University of Casablanca; National Higher School of Art and Design (ENSAD), Hassan II University of Casablanca
- Mohamed AZZOUAZI — LTIM – Laboratory of Information Technology and Modeling, Faculty of Sciences Ben M'Sik (FSBM), Hassan II University of Casablanca

Correspondence: **younes.nadir@univh2c.ma**

## Dataset

The study uses the **Breast Cancer Wisconsin (Diagnostic) (WDBC)** dataset.

- Source: UCI Machine Learning Repository
- DOI: `10.24432/C5DW2B`
- Programmatic loader used by the analysis: `sklearn.datasets.load_breast_cancer()`

The dataset is **not redistributed** in this repository. It is loaded programmatically by the analysis code.

## Study design reproduced by this repository

The analysis evaluates:

- Logistic Regression;
- radial-basis-function Support Vector Machine (SVM);
- Random Forest;
- XGBoost;
- Soft Voting Ensemble.

The evaluation pipeline includes:

- stratified 80/20 train-test split;
- 5-fold × 3-repeat stratified cross-validation on the training partition;
- fixed held-out test evaluation;
- 5,000-resample nonparametric bootstrap confidence intervals;
- ROC-AUC, PR-AUC, accuracy, precision, sensitivity, specificity, NPV, F1, balanced accuracy, MCC, and Brier score;
- calibration curves;
- all-pair exact McNemar tests;
- all-pair DeLong ROC-AUC comparisons;
- Holm correction for multiple comparisons;
- permutation importance for the soft-voting ensemble;
- TreeSHAP-based feature importance for XGBoost;
- held-out error analysis;
- computational stress testing using synthetically expanded training sets.

## Repository structure

```text
breast-cancer-reproducible-ml/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── SHA256SUMS.txt
├── code/
│   ├── canonical_analysis.py
│   ├── all_pairwise_tests.py
│   ├── delong_compare.py
│   └── reproduce_all.py
├── environment/
│   ├── requirements.txt
│   ├── environment.yml
│   └── environment.json
├── results/
│   ├── model_performance_table.csv
│   ├── repeated_cv_fold_results.csv
│   ├── calibration_curve_points.csv
│   ├── all_pairwise_tests.csv
│   ├── misclassified_cases.csv
│   ├── ensemble_permutation_importance.csv
│   ├── xgboost_shap_importance.csv
│   ├── stress_test_results.csv
│   ├── stress_test_raw.csv
│   ├── dataset_description_table.csv
│   └── data_quality_report.csv
└── figures/
    ├── Figure1_Workflow.png
    ├── Figure2_ROC_and_Calibration.png
    ├── Figure3_Soft_Voting_Confusion_Matrix.png
    ├── Figure4_Explainability_Permutation_and_TreeSHAP.png
    └── Figure5_Computational_Stress_Test.png
```

## Recommended environment

The canonical environment recorded for the manuscript is:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- XGBoost 3.1.3
- SciPy 1.17.0
- Matplotlib 3.10.8
- statsmodels 0.14.5
- psutil 7.0.0

### Option 1 — Conda

```bash
conda env create -f environment/environment.yml
conda activate breast-cancer-benchmark
python code/reproduce_all.py
```

### Option 2 — pip

Create and activate a Python 3.13 environment, then run:

```bash
python -m pip install -r environment/requirements.txt
python code/reproduce_all.py
```

## Reproducing the complete analysis

From the repository root:

```bash
python code/reproduce_all.py
```

The script runs the canonical core analysis, the all-pair statistical comparisons, and the computational stress test. Newly generated files are written to:

```text
code/generated_outputs/
```

The `results/` directory contains the **canonical outputs used to support the manuscript**. This separation allows a reviewer to compare newly reproduced outputs against the archived canonical results.

## Reproducibility settings

- Random seed: `42`
- Held-out test fraction: `0.20`
- Cross-validation: `5` folds × `3` repeats
- Bootstrap resamples: `5000`
- Stress-test repetitions: `3`
- Single-threaded fitting is used where specified to reduce nondeterministic numerical variation.

Minor timing differences are expected across hardware and operating systems. The computational stress test should therefore be interpreted in terms of relative scaling behavior rather than exact wall-clock equality across machines.

## Canonical results and integrity

`SHA256SUMS.txt` contains SHA-256 hashes for the versioned repository files. To verify them on Linux/macOS from the repository root:

```bash
sha256sum -c SHA256SUMS.txt
```

On macOS systems where `sha256sum` is unavailable, an equivalent SHA-256 verification utility may be used.

## Figures

The `figures/` directory contains the exact composite figures used in the submitted manuscript. The underlying numerical information is also available in `results/`, and the canonical analysis produces the component plots in `code/generated_outputs/`.

## Data and code availability

The WDBC dataset is publicly available from UCI and is loaded programmatically; it is not included in this repository. Analysis code, pinned environment specifications, canonical numerical outputs, and manuscript figures are provided here for reproducibility.

For a permanent archival record, the recommended workflow is to create a GitHub release (for example `v1.0.0`) and archive that release in Zenodo to obtain a DOI. The resulting repository URL and Zenodo DOI can then be added to the manuscript's Data Availability Statement.

## Generative AI transparency

As reported in the manuscript, ChatGPT (OpenAI) was used under author supervision for language refinement, consistency checking, and review/refactoring support for analysis code. Experimental observations, labels, model outputs, statistical results, and figures were generated by the documented computational workflow and reviewed by the authors.

## Citation

A `CITATION.cff` file is included so that GitHub can expose a **Cite this repository** action. Once the associated article has been published, update `CITATION.cff` with the final article DOI and bibliographic information.

## License

The analysis code and repository-authored documentation are released under the MIT License. This license does not apply to the externally hosted WDBC dataset, which remains subject to the terms of its original source.
