# Netflix Show Success Prediction - End-to-End Data Analysis Project

## Objective

The goal of this project is to **predict whether a Netflix show is likely to be successful** using a range of metadata features such as language, country, genre, popularity, and more. This is a classification problem designed to simulate real-world business decision-making in the streaming industry.

---

## Why This Project?

As Netflix continues to produce and acquire global content, understanding what makes a show successful is key to optimizing production investments. By analyzing past show data, we aim to **train a machine learning model to predict show success**, which can guide strategic planning for content investments.

---

## Data Processing & Feature Engineering

- Dataset: Netflix Shows (2025 snapshot)
- Target variable `success` was **engineered using the `rating` column** (not `vote_average`), where shows were labeled as successful if their rating exceeded a defined threshold.
- Handled multivalued columns (like `genres`) by exploding them and applying one-hot encoding.
- Cleaned and encoded categorical variables (`country`, `language`, `genre`) using one-hot encoding.

---

## Exploratory Data Analysis (EDA)

- Identified genre-wise success rates — e.g., **Sci-Fi & Fantasy** and **Action & Adventure** genres showed varying success patterns.
- Found **'vote_count'** to be the strongest predictor-9.
- Analyzed distribution and outliers in features like `popularity`, `release_year`, and `rating`.

---

## Model Building & Experiments

### Phase 1: Decision Tree Classifier  
- Initially included `vote_average` and achieved **100% accuracy** — clear **overfitting** due to data leakage.
- Removed `vote_average` and re-trained — performance dropped significantly but became realistic.

### Phase 2: Feature Importance  
- Identified most predictive features: `vote_count`, `popularity`, `release_year`, `language`, and top genres/countries.

### Phase 3: Random Forest Classifier  
- Built and tuned two models:
  - **Model A**: Included all features (incl. `vote_average`) → Overfit (100% accuracy, unreliable)
  - **Model B**: Included only important, clean features → More generalizable

---

## Hyperparameter Tuning & Cross-Validation

- Used `RandomizedSearchCV` to optimize parameters (`n_estimators`, `max_depth`, etc.).
- Performed cross-validation using:
  - All features: F1 Score ≈ **0.5538**, Accuracy ≈ **0.6559**
  - Important features only: F1 Score ≈ **0.5242**, Accuracy ≈ **0.6179**

---

## Final Model Evaluation (on test set)

| Metric             | Value     |
|--------------------|-----------|
| Accuracy           | 0.6825    |
| Precision          | 0.6655    |
| Recall             | 0.5393    |
| F1 Score           | 0.5958    |
| AUC Score          | 0.7317    |

- Confusion Matrix:
  - True Positives: 577
  - True Negatives: 1106
  - False Positives: 290
  - False Negatives: 493

---

## Key Visualizations

- Feature Importances bar chart
- Confusion Matrix heatmap
- ROC Curve with AUC
- Genre-wise success rate
- Class distribution before and after feature engineering

---

## Tech Stack

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- Jupyter Notebooks

---

## Lessons Learned

- Data leakage is subtle — features like `vote_average` closely tied to the label can inflate metrics falsely.
- Feature selection and domain logic are as important as model complexity.
- Cross-validation is essential for evaluating model generalizability.

---

## Author
- Shijin Ramesh, Data Analyst
