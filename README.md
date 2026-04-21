# ADHD Prediction Using NLP

A machine learning pipeline that predicts whether a university student is likely to have ADHD using Natural Language Processing (NLP) and structured psychometric data.

---

## What This Project Does

This project takes responses from standardised psychological questionnaires (ASRS, BDI, BAI, AUDIT, AAS) and a free-text diagnosis field, and trains five machine learning models to predict ADHD likelihood. The best model (Logistic Regression) achieves **81.4% accuracy** with an **AUC of 0.881**.

---

## Project Structure

```
adhd_project/
├── data/
│   └── ADHD.xlsx                   ← Put your dataset here
├── src/
│   ├── data_loader.py              ← Loads data, creates target variable
│   ├── feature_engineering.py      ← TF-IDF, feature selection, encoding
│   ├── model.py                    ← All 5 model definitions + save/load
│   └── evaluate.py                 ← Metrics, confusion matrix, ROC curves
├── models/                         ← Trained .pkl files saved here automatically
├── notebooks/
│   └── 02_model_training.ipynb     ← Main notebook — run this
├── requirements.txt                ← All required libraries
└── README.md                       ← This file
```

---

## Requirements

- Python 3.9 or higher
- Windows / Mac / Linux
- VS Code (recommended) or any text editor
- ~500 MB free disk space

---

## Step-by-Step Setup Guide

### Step 1 — Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/NeerajSh-16/adhd_project.git
cd adhd_project
```

If you do not have Git installed, download the ZIP from GitHub:
- Click the green **Code** button on the repository page
- Click **Download ZIP**
- Extract the ZIP and open a terminal inside the extracted folder

---

### Step 2 — Add the dataset

Place the `ADHD.xlsx` file inside the `data/` folder:

```
adhd_project/
└── data/
    └── ADHD.xlsx    ← file goes here
```

> **Note:** The dataset is not included in the repository due to privacy considerations. Contact the repository owner to obtain it.

---

### Step 3 — Create a virtual environment

A virtual environment keeps all project libraries isolated from your system Python. Everything installs inside the `venv/` folder and can be removed cleanly later.

```bash
python -m venv venv --without-pip
```

> **Windows users:** If this command fails, it may be interrupted by antivirus software. Try running the terminal as Administrator, or use the `--without-pip` flag shown above.

---

### Step 4 — Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You will see `(venv)` appear at the start of your terminal line when it is active. You must activate the virtual environment every time you open a new terminal window.

---

### Step 5 — Install required libraries

```bash
pip install -r requirements.txt
```

This installs all dependencies listed in `requirements.txt`. It will take 2–3 minutes on first run.

Then install XGBoost separately:

```bash
pip install xgboost
```

---

### Step 6 — Register the environment with Jupyter

This tells Jupyter to use your project's Python instead of the system Python:

```bash
python -m ipykernel install --user --name=adhd_venv --display-name "Python (adhd_venv)"
```

---

### Step 7 — Verify the installation

Run this quick check to confirm everything is installed correctly:

```bash
python -c "import pandas; import sklearn; import scipy; import xgboost; print('All libraries installed successfully!')"
```

You should see:
```
All libraries installed successfully!
```

If you see an error mentioning a missing library, install it manually:
```bash
pip install <library-name>
```

---

### Step 8 — Launch Jupyter Notebook

```bash
cd notebooks
jupyter notebook
```

Your browser will open automatically showing the Jupyter interface.

---

### Step 9 — Open the notebook and select the kernel

1. Click on `02_model_training.ipynb` to open it
2. In the top-right corner, click the kernel name
3. Select **"Python (adhd_venv)"** from the dropdown list

> **Important:** If you do not select the correct kernel, the notebook will not find the installed libraries.

---

### Step 10 — Run the notebook

In the Jupyter menu, click:

```
Kernel → Restart & Run All
```

This runs every cell from top to bottom automatically. The full pipeline takes approximately 2–3 minutes to complete.

---

## What the Notebook Does (Cell by Cell)

| Cell | What it does |
|------|-------------|
| Cell 1 | Sets up auto-reload and adds `src/` to Python path |
| Cell 2 | Loads `ADHD.xlsx`, creates the target variable (ASRS Part A >= 14), handles missing values |
| Cell 3 | Builds all features: TF-IDF text features + numeric scale scores + encoded categoricals. Checks class balance |
| Cell 4 | Applies Mutual Information feature selection — keeps best 20 features from 124 |
| Cell 5 | Trains all 5 models and saves them to `models/` as `.pkl` files |
| Cell 6 | Evaluates each model — prints accuracy, F1, recall, precision, and confusion matrix |
| Cell 7 | Plots ROC curves for all models on one chart |
| Cell 8 | Runs 5-fold cross validation for reliable performance estimates |
| Cell 9 | Prints final comparison table of all models sorted by accuracy |

---

## Expected Results

After running the notebook you should see results close to these:

| Model | Accuracy | F1 Score | Recall | Precision | AUC |
|-------|----------|----------|--------|-----------|-----|
| Logistic Regression | 81.4% | 80.8% | 78.4% | 83.3% | 0.881 |
| SVM (Linear) | 81.4% | 80.4% | 76.5% | 84.8% | 0.888 |
| Stacking (LR + XGB) | 81.4% | 80.0% | 74.5% | 86.4% | 0.862 |
| XGBoost | 75.5% | 75.2% | 74.5% | 76.0% | 0.812 |
| MLP Neural Network | 69.6% | 69.3% | 68.6% | 70.0% | 0.754 |

> Small variations (±1–2%) are normal due to randomness in model initialisation. The general ranking should remain the same.

---

## How to Run Again After Closing VS Code

Every time you return to work on this project:

```bash
# 1. Navigate to the project folder
cd adhd_project

# 2. Activate the virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Launch Jupyter
cd notebooks
jupyter notebook
```

Then open `02_model_training.ipynb`, ensure the kernel is set to **Python (adhd_venv)**, and run:
```
Kernel → Restart & Run All
```

---

## Files Generated After Running

After a successful run, the `models/` folder will contain:

```
models/
├── logistic_regression.pkl             ← Saved Logistic Regression model
├── svm.pkl                             ← Saved SVM model
├── xgboost.pkl                         ← Saved XGBoost model
├── mlp.pkl                             ← Saved MLP model
├── stacking.pkl                        ← Saved Stacking model
├── Logistic_Regression_cm.png          ← Confusion matrix plot
├── SVM_(Linear)_cm.png
├── XGBoost_cm.png
├── MLP_cm.png
├── Stacking_(LR+XGB)_cm.png
└── roc_curves_all_models.png           ← ROC curve comparison chart
```

---

## Target Variable Explained

The model predicts **ASRS Part A >= 14** as the target:

- **ASRS** = Adult ADHD Self-Report Scale (WHO / Harvard Medical School, Kessler et al. 2005)
- **Part A** = Questions 1–6, the clinically validated ADHD screener
- **Score >= 14** = Positive screen for ADHD (90% sensitivity, 88% specificity)
- **Class balance** = 251 positive (49.6%) vs 255 negative (50.4%) — perfectly balanced

---

## Feature Selection Explained

- **Total features built:** 124 (22 numeric + 4 categorical + up to 100 TF-IDF)
- **Algorithm:** SelectKBest with Mutual Information (`mutual_info_classif`)
- **k selected:** 20 — tested values from 20 to 124, k=20 gave best accuracy (81.4%)
- **Features kept:** ASRS Part B items, BAI scores, BDI total + item 19, AAS items 3/4/6, categorical variables, top TF-IDF words

---

## Common Errors and Fixes

**Error: `ModuleNotFoundError: No module named 'xgboost'`**
```bash
# Make sure venv is active, then:
pip install xgboost
```

**Error: `FileNotFoundError: Could not find the file at: ../data/ADHD.xlsx`**
```
Make sure ADHD.xlsx is placed inside the data/ folder.
Check the filename is exactly ADHD.xlsx (capital letters, no spaces).
```

**Error: Kernel not found / wrong kernel selected**
```bash
# Re-register the kernel:
python -m ipykernel install --user --name=adhd_venv --display-name "Python (adhd_venv)"
# Then restart Jupyter and select "Python (adhd_venv)" again
```

**Error: `venv\Scripts\activate` not recognised on Windows**
```bash
# Try this instead:
.\venv\Scripts\activate
# Or open PowerShell as Administrator and run:
Set-ExecutionPolicy RemoteSigned
```

**Library installed but notebook still can't find it**
```
You likely installed the library without the venv active.
Activate venv first, then reinstall:
venv\Scripts\activate
pip install <library-name>
```

---

## How to Remove Everything

To completely remove the project and all installed libraries:

1. Delete the `venv/` folder (this removes all installed packages)
2. Delete the `adhd_project/` folder (this removes all project files)

No traces are left on your system — everything was contained inside these two folders.

---

## Source Files Overview

| File | Purpose |
|------|---------|
| `src/data_loader.py` | `load_data()` — reads Excel file. `create_target()` — builds ASRS Part A binary label. `handle_missing()` — fills NaN values |
| `src/feature_engineering.py` | `build_feature_matrix()` — combines all features. `check_class_imbalance()` — verifies balance. `apply_feature_selection()` — runs Mutual Info k=20 |
| `src/model.py` | `train_logistic_regression()`, `train_svm()`, `train_xgboost()`, `train_mlp()`, `train_stacking()`. Also `save_model()` and `load_model()` |
| `src/evaluate.py` | `evaluate_model()` — prints all metrics. `plot_confusion_matrix()` — saves confusion matrix image. `plot_roc_curves()` — saves ROC chart. `cross_validate_all()` — 5-fold CV. `compare_models()` — final summary table |

---

## References

- Kessler, R. C., et al. (2005). The World Health Organization Adult ADHD Self-Report Scale (ASRS). *Psychological Medicine, 35*(2), 245–256.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

---

## Contact

Repository: https://github.com/NeerajSh-16/adhd_project
