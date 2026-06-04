project predictive maitenace
dataset from kaggel
---

##  Problem Statement

Industrial machines operate under varying thermal, mechanical, and rotational conditions.
The goal is to predict machine failure before it occurs using sensor readings so that
maintenance teams can intervene proactively and avoid unplanned downtime. The dataset is labled and model use this knowelage to learn patterns inside data,
and then predict probbality of failure. Alos new columns are computed from alredy available ones- e.g. Power
.pptx slides are attached to support description of this file

In this context a **false negative** (missed failure) is significantly more costly than
a **false positive** (unnecessary inspection) — this asymmetry drives all modelling decisions.
---
## Models
Three models are trained and evaluated:

| Model | Imbalance handling | even creating weight or systetic failures was not option here since dataste is too perfect
|---|---|
| XGBoost (calibrated) | `scale_pos_weight` + `CalibratedClassifierCV` |
| Random Forest | `class_weight="balanced_subsample"` |
| Logistic Regression | `class_weight="balanced"` + L1 penalty |

## XGBoost drives the dashboard visuals. Random Forest and Logistic Regression run in background.
Primary metric is **Recall** — catching real failures is the priority.
---
## How to Run
**1. Install dependencies**
```bash
pip install -r requirements.txt
```
**2. Train models**
```bash
python train_model.py
```
**3. Run the dashboard**
```bash
streamlit run app.py / or available via URL 
```
**4. Run tests**
```bash
pytest test_train_model.py -v
```
---
##  Notebooks
`notebooks/dataset_checks.ipynb` contains exploratory data analysis including:
- Label consistency check between `Target` and `Failure Type` columns
- ~0.18% of rows contain an actual failure type but are labelled `Target = 0`
- AI-powered data quality audit using Groq LLaMA 3.3 (null rates, duplicates, outliers)

#### This notebook is for analysis only and is not part of the production pipeline.
---
##  Links

- [Streamlit Dashboard]
- ([https://your-app-url.streamlit.app](https://ipapplutionpredictivemaintenance-hcyrqtiup35e3mhgljkbdv.streamlit.app/))
- [GitHub Repository](https://github.com/mila151107/IPS_solution_predictive_maintenance)
- [Dataset — Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
