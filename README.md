# Material Properties ML - Predicting Band Gap with Machine Learning

A machine learning project for predicting the band gap of materials based on structural features.

## Project Description

Band gap is a property that determines whether a material is a metal, semiconductor, or insulator. It plays a key role in electronics, solar cells, and the design of new materials.

Data was sourced from the [Materials Project](https://materialsproject.org), wchich is a free materials database.

### Why do we exclude metals?

Over 71% of materials in the database are metals with a band gap of 0. Including them causes the model to learn that predicting zero is almost always a safe bet, achieving seemingly good results while being practically useless. By filtering to materials with band gap > 0 (semiconductors and insulators), we get a cleaner signal and a more meaningful model.

---

## Project Pipeline

1. **EDA** - data distributions, correlation matrix, missing value analysis
2. **Feature Engineering** - filtering metals, normalization (StandardScaler), 80/20 split
3. **Models** - Linear Regression, Random Forest, Gradient Boosting
4. **Evaluation** - RMSE, Cross-Validation (5-fold)
5. **Interpretability** - Feature Importance

---

## Technologies Used

- **Python 3**
- **Pandas** - data manipulation
- **NumPy** - numerical computing
- **Matplotlib / Seaborn** - data visualization
- **Scikit-learn** - machine learning models
- **mp-api** - fetching data from Materials Project

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/xJadzix/material-properties-ml.git
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn mp-api
```

3. Get a free API key at [materialsproject.org](https://materialsproject.org) and paste it into the notebook.

4. Open `Material_Properties_ML.ipynb` in Jupyter Notebook and run the cells in order.

---

## Model Results

Models were trained on 2813 materials (semiconductors and insulators) with an 80/20 train-test split.

| Model | RMSE | CV RMSE (5-fold) |
|---|---|---|
| Linear Regression | 1.7699 | 1.7898 |
| Random Forest | 1.6118 | 1.7219 |
| **Gradient Boosting** | **1.5944** | **1.7386** |

RMSE values are in eV.

### Why do we exclude Linear Regression from the final conclusions?

EDA showed that no single feature correlates strongly with band gap (the highest correlation was -0.29 for density). This indicates that the relationships in the data are **non-linear** — and Linear Regression assumes linearity, making it unable to capture these patterns. Random Forest and Gradient Boosting handle non-linear relationships much better, as confirmed by the RMSE results.

### Feature Importance

Average feature importance across Random Forest and Gradient Boosting:

| Feature | Importance |
|---|---|
| density | 0.350 |
| volume | 0.337 |
| nsites | 0.313 |

All three features have similar importance - none dominates, so none should be dropped.

---

## Possible Improvements

+ Adding chemical features (elemental composition, electronegativity)
+ Larger dataset
+ Hyperparameter tuning (GridSearchCV)
+ Using XGBoost or LightGBM
