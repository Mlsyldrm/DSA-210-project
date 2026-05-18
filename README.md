# DSA210_TermProject

# Skincare & Ethics: Does Cruelty-Free Status Affect Price, Popularity, and Ratings?

**DSA 210 - Introduction to Data Science**  
**Sabancı University**  
**Student/ID: Melisa Ece Yıldırım / 32053**

---

🌐 **Project Presentation:**  
👉

📄 This repository contains the full technical implementation, dataset, statistical tests, and machine learning models for the project.  
The website serves as a high-level presentation, while all details are documented here.

---

## Project Overview

This project explores whether the **cruelty-free (CF) status** of skincare products is significantly associated with their **price**, **popularity (loves_count)**, and **user ratings**, using a merged dataset of 293 Sephora skincare products. It aims to understand whether ethical positioning translates into a measurable market footprint — and to evaluate, through machine learning, whether product attributes can actually predict these outcomes.

**Central Question:**

> Does being a cruelty-free skincare product translate into a measurable difference in price, popularity, or user satisfaction — and how well can product attributes predict these outcomes?

---

## Motivation

As a consumer who pays attention to ethical considerations when purchasing personal care products, I have always wondered whether being "cruelty-free" (not tested on animals) actually translates into a measurable difference in the marketplace. The skincare industry has grown rapidly in recent years, and consumer demand for ethically positioned products has risen along with it.

<details>
<summary><b>Why This Topic Matters?</b></summary>

1. **Ethics vs. Economics:**  
   Does ethical positioning carry a price premium? Or is the cruelty-free label primarily a marketing differentiator that doesn't manifest in actual price differences?

2. **Consumer Behavior:**  
   Are consumers willing to "reward" cruelty-free brands with stronger engagement (higher loves_count) or better ratings? Or does ethics remain a secondary concern after price and perceived quality?

3. **Predictability of Success:**  
   Beyond statistical relationships, can a machine learning model use product attributes (price, popularity, skin-type compatibility) to predict ratings or even the CF status of a product? Where does product-level data fall short?

4. **Limits of Observational Data:**  
   This project also serves as a methodological exercise: what happens when correlations are statistically significant but practically weak? When narrow target variance kills predictive power? Understanding _when_ models fail is as important as building ones that work.

### Practical Relevance

- **For Consumers:** Are you really paying more (or less) for cruelty-free skincare?
- **For Brands:** Does CF certification carry a measurable popularity boost?
- **For Researchers:** A case study in interpreting weak correlations and narrow-variance regression problems.

</details>

These questions guide the project to investigate:

1. How **price, popularity, and ratings** relate to one another
2. Whether **cruelty-free status** shifts any of these variables in a statistically meaningful way
3. The **predictive limits** of product-level features through four ML models

---

## Main Findings (Summary)

| Hypothesis                             | Result           | p-value | Key Insight                                     |
| -------------------------------------- | ---------------- | ------- | ----------------------------------------------- |
| **H1:** Price → Rating (negative)      | ✅ **Supported** | 0.007   | Weak negative correlation (ρ = −0.143)          |
| **H2:** Price → Popularity (negative)  | ✅ **Supported** | <0.001  | Strongest association in the study (ρ = −0.387) |
| **H3:** Popularity → Rating (positive) | ✅ **Supported** | 0.005   | Weak positive correlation (ρ = +0.150)          |
| **H4:** CF status ↔ Rating             | ❌ Not Supported | 0.357   | No meaningful difference between CF and non-CF  |
| **H5:** CF status → Lower Price        | ❌ Not Supported | 0.086   | Marginal, but not statistically significant     |

**Overall Score:** 3/5 hypotheses supported (60%)

**Note:** All supported correlations are _statistically_ significant but _practically_ weak. The strongest finding is the price–popularity relationship; cruelty-free status itself shows no measurable footprint on price or rating.

<details>
<summary><b>📈 Click to see detailed results</b></summary>

### H2: Price ↔ Popularity ✅ (Strongest)

- **Spearman ρ:** −0.387 (p < 0.001)
- **Pearson r:** −0.352 (p < 0.001)
- **Interpretation:** Cheaper products tend to attract more "loves" — but this is an association, not causation.

### H1: Price ↔ Rating ✅

- **Spearman ρ:** −0.143 (p = 0.007)
- **Effect size:** Very small; ratings are compressed in the 4.0–4.6 range.

### H3: Popularity ↔ Rating ✅

- **Spearman ρ:** +0.150 (p = 0.005)
- **Interpretation:** Popular products are _slightly_ more likely to receive higher ratings.

### H4: CF ↔ Rating ❌

- **Mann-Whitney U:** 8970 (p = 0.357)
- **Group medians:** CF=0 ≈ 4.30, CF=1 ≈ 4.30 → indistinguishable

### H5: CF → Lower Price ❌

- **Mann-Whitney U:** 10540 (p = 0.086)
- **Conclusion:** Boxplot suggests slightly lower prices for CF products, but not significant.

**Key Finding:** Cruelty-free positioning carries no measurable market footprint on price or rating. The dominant signal in the data is the negative price–popularity relationship.

</details>

---

## 📊 Dataset

### Dataset Size

- **Total products:** 293
- **Cruelty-free (CF=1):** 99 products (33.8%)
- **Non-cruelty-free / unknown (CF=0):** 194 products (66.2%)

### Variables

<details>
<summary><b>View complete variable list</b></summary>

**Product-level features:**

- `brand`, `product_name`, `label` (product category)
- `price` (USD), `log_price` (log-transformed)
- `loves_count` (popularity proxy), `log_loves_count`
- `rating` (1–5 scale)

**Binary skin-type compatibility:**

- `dry`, `normal`, `oily`, `sensitive`

**Engineered features:**

- `CF` (binary: cruelty-free brand flag)
- `skin_compatibility_score` (sum of skin-type flags)

</details>

## Data Sources

| Data                       | Source                          | Format |
| -------------------------- | ------------------------------- | ------ |
| Sephora skincare catalogue | Kaggle (public Sephora dataset) | CSV    |
| Cruelty-free brand list    | Public CF brand reference list  | CSV    |

<details>
<summary><b>Note on brand-level CF labeling:</b></summary>

Cruelty-free status was assigned at the **brand level** using a public reference list rather than from official certifications (Leaping Bunny, PETA). Brands not appearing on this list were treated as non-CF (CF=0). This labeling strategy may overstate the non-CF group and introduce some label noise; future work could integrate official certification databases.

</details>

---

## Methodology

### Preprocessing

- **Merging:** The Sephora product dataset was merged with the cruelty-free brand list on the `brand` field. Brands found on the list were flagged as CF=1; the rest as CF=0.
- **Log transformations:** Both `price` and `loves_count` were heavily right-skewed. Log transformations were applied to bring distributions closer to normal — necessary for valid use of correlation tests and linear models.
- **Feature engineering:** A composite `skin_compatibility_score` was created by summing the four skin-type binary flags.
- **No imputation needed:** Key analysis columns (price, rating, loves_count) had no missing values.

### Statistical Methods

<details>
<summary><b>H1: Price → Rating Test</b></summary>

**Method:** One-sided Spearman correlation  
**Variables:** `log_price` vs `rating`

**Rationale:** Spearman (rather than Pearson) is used because rating is concentrated in a narrow band (4.0–4.6) and the relationship is not assumed linear.

**Results:**

- ρ = −0.143
- p-value = 0.007 (one-sided)
- **H₀ rejected** — weak but significant negative relationship.

</details>

<details>
<summary><b>H2: Price → Popularity Test</b></summary>

**Method:** One-sided Spearman correlation  
**Variables:** `log_price` vs `log_loves_count`

**Results:**

- ρ = −0.387
- p-value < 0.001
- Pearson r = −0.352 (consistent direction)
- **H₀ rejected** — strongest relationship in the analysis.

**Interpretation:** Cheaper products tend to be more popular. Could reflect accessible pricing strategies of popular brands rather than direct causal effect.

</details>

<details>
<summary><b>H3: Popularity → Rating Test</b></summary>

**Method:** One-sided Spearman correlation  
**Variables:** `log_loves_count` vs `rating`

**Results:**

- ρ = +0.150
- p-value = 0.005 (one-sided)
- **H₀ rejected** — weak but significant positive relationship.

</details>

<details>
<summary><b>H4: CF Status ↔ Rating Test</b></summary>

**Method:** Two-sided Mann-Whitney U test  
**Groups:** CF=0 (n=194) vs CF=1 (n=99)

**Rationale:** Non-parametric test used because rating distributions are non-normal and group sizes are unbalanced.

**Results:**

- U-statistic = 8970
- p-value = 0.357
- **Failed to reject H₀** — no significant difference in ratings between CF and non-CF products.

</details>

<details>
<summary><b>H5: CF → Lower Price Test</b></summary>

**Method:** One-sided Mann-Whitney U test  
**Hypothesis direction:** CF products are _less expensive_ than non-CF.

**Results:**

- U-statistic = 10540
- p-value = 0.086
- **Failed to reject H₀** — marginal but not significant.

**Interpretation:** Boxplot shows CF=0 has slightly higher prices, but the difference is not statistically convincing at α = 0.05.

</details>

---

## 📈 Key Visualizations

### EDA Analysis

- **Price distributions:** Raw vs log-transformed (justifying log usage)
- **Loves_count distributions:** Raw vs log-transformed
- **CF and skin-type counts:** Showing class imbalances
- **Scatter plots:** All pairwise relationships between log_price, log_loves_count, and rating
- **Boxplots by CF status:** Visual comparison of rating, price, and popularity across groups
- **Spearman correlation heatmap:** Summary of all numeric variable relationships

### Hypothesis Test Results

- All five hypothesis tests are visualized with boxplots or regression scatterplots in the notebook and the presentation website.

See [`Skincare_ethics_analysis.ipynb`](Skincare_ethics_analysis.ipynb) for complete EDA and test figures.

---

## Machine Learning Analysis ✅

This project extends the statistical findings with four ML experiments to test whether product attributes can actually predict ratings, popularity, category, and CF status.

### Four Models Tested:

1. **Linear Regression (Rating)** — Predict `rating` from `log_price`, `log_loves_count`, `skin_compatibility_score`
2. **Linear Regression (Popularity)** — Predict `log_loves_count` from price, rating, CF status
3. **Random Forest Classifier (Category)** — Predict product `label` from numeric features
4. **Logistic Regression (CF status)** — Predict `CF` from product attributes

### Key Findings:

| Model               | Target          | Best Performance            | Conclusion                      |
| ------------------- | --------------- | --------------------------- | ------------------------------- |
| Linear Regression   | rating          | CV R² = −0.019              | ❌ Worse than mean baseline     |
| Linear Regression   | log_loves_count | CV R² = 0.140               | ⚠️ Explains 14% of variance     |
| Random Forest       | category        | Accuracy = 45.8%, F1 = 0.43 | ⚠️ Moderate, GridSearchCV-tuned |
| Logistic Regression | CF status       | CV F1 = 0.469, AUC = 0.606  | ⚠️ Marginal predictive signal   |

**Result:** Rating is essentially unpredictable from these features — its variance is too narrow (95% of ratings fall between 4.0 and 4.6). Popularity is moderately predictable, and the strongest predictor is consistently `log_price` (negative β), aligning perfectly with the H2 finding.

<details>
<summary><b>📈 Click to see detailed ML results</b></summary>

### Model 1: Linear Regression for Rating

- Single train-test split: R² = −0.231 (worse than predicting the mean!)
- After 5-fold CV with StandardScaler: R² = −0.019, RMSE = 0.377
- **Conclusion:** Rating variance is too compressed to be modeled with these features.

### Model 2: Linear Regression for Popularity

- CV R² = 0.140 (positive!)
- Strongest coefficient: log_price (β = −0.435) → confirms the H2 finding
- **Conclusion:** Modest but meaningful — coefficient directions align with hypotheses.

### Model 3: Random Forest Classifier for Product Category

- Baseline accuracy: 42.4%
- After GridSearchCV (best: max_depth=10, n_estimators=200): **45.8% accuracy, F1 = 0.43**
- "Face Mask" F1 improved from 0.00 → 0.20
- "Sun protect" remained weakest (only 5 test samples)

### Model 4: Logistic Regression for CF Status

- GridSearchCV best params: C = 0.01, solver = liblinear
- Strong regularization indicates weak predictive signal in available features
- CV F1 = 0.469, AUC = 0.606
- Strongest predictor: log_loves_count (consistent with CF products being slightly more popular)

**Key Takeaway:** All four models converge on the same story — product-level numeric features carry limited predictive signal, especially for the CF status itself. The strongest pattern in the data is the price–popularity relationship.

</details>

---

## Key Insights

### 1. **Price–Popularity is the Dominant Signal** ⭐

The negative relationship between price and popularity (ρ = −0.387) is the strongest finding in the entire analysis — far stronger than any rating-related relationship. Accessible pricing appears to drive consumer engagement more than any other observable factor in this dataset.

### 2. **Cruelty-Free Status Has No Measurable Footprint**

Cruelty-free products are neither significantly more expensive nor significantly cheaper, and they don't receive different ratings. The ethical label, in this dataset, does not translate into a quantifiable market difference.

### 3. **Ratings Are Structurally Compressed**

95% of product ratings fall between 4.0 and 4.6. This is a known feature of e-commerce review systems (selection bias, social desirability), but it cripples regression: the target barely moves, so even meaningful features cannot explain its variance. **Statistical significance ≠ practical importance.**

### 4. **ML Confirms the Statistical Story**

All four ML models — across different targets and algorithms — converge on the same picture: weak product-level signal, especially for predicting CF status itself. This is not a failure of the models; it's a finding about the data.

---

## Technologies & Tools

**Core Libraries:**

- `Python 3.9+`
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scipy` — statistical tests (Spearman, Mann-Whitney U, Pearson)
- `scikit-learn` — machine learning

**Machine Learning:**

- Linear Regression, Logistic Regression
- Random Forest Classifier
- K-Nearest Neighbors (comparison baseline)
- GridSearchCV for hyperparameter tuning
- 5-Fold Cross Validation, StandardScaler

**Development:**

- `jupyter` — interactive notebooks
- `git` — version control

---

## Reproducibility

All analyses are fully reproducible:

1. Merged dataset available at `data/processed/merged_cosmetic_product_dataset.csv`
2. Main notebook: `Skincare_ethics_analysis.ipynb` (complete pipeline)
3. Fixed random seeds (`random_state=42`) for all stochastic models
4. Hyperparameter grids documented inline

<details>
<summary><b>Installation & Setup</b></summary>

```bash
# Clone repository
git clone https://github.com/<your-username>/DSA210_TermProject
cd DSA210_TermProject

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Skincare_ethics_analysis.ipynb
```

**Requirements:**

- Python 3.9+
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn
- jupyter

</details>

---

## Project Contributions

This analysis offers value in three areas:

1. **Empirical Test of Ethical Pricing**  
   Provides concrete evidence on whether cruelty-free positioning carries a measurable market premium in mainstream skincare retail.

2. **A Case Study in Weak Effects**  
   Demonstrates the gap between statistical significance and practical importance — three hypotheses are formally supported, but all effect sizes are weak.

3. **Methodological Honesty in ML**  
   Shows what happens when ML models meet a low-signal feature space: negative R², highly regularized classifiers, and convergence on simple baselines. A useful counterexample for "ML can predict anything" intuitions.

---

## Limitations & Future Work

**Current Limitations:**

- **Small and imbalanced sample:** 293 products (99 CF vs 194 non-CF) limits statistical power and constrains classifiers.
- **Single-retailer bias:** All product data comes from Sephora. Results may not generalize to drugstore brands or international markets.
- **Brand-level CF labeling:** CF status was assigned using a public brand list, not official certifications (Leaping Bunny, PETA). This may introduce label noise.
- **Compressed rating variance:** 95% of ratings fall between 4.0 and 4.6, making rating prediction nearly impossible.
- **Snapshot in time:** Single data collection moment — no longitudinal information about pricing or popularity changes.
- **Limited feature set:** Ingredient lists, packaging, marketing spend, brand age, and review text were not available.
- **Association ≠ causation:** Every finding is an observed relationship, not a causal effect.
- **No demographic context:** Nothing about who buys or rates these products.

**Future Extensions:**

- Incorporate **official CF certification databases** (Leaping Bunny, PETA Beauty Without Bunnies) for cleaner labeling
- Expand to **multiple retailers** (Ulta, drugstore, e-commerce platforms) for broader generalizability
- Add **NLP on product descriptions and ingredient lists** for richer feature representation
- **Longitudinal data collection** to observe how cruelty-free positioning, price, and popularity evolve together
- **Review text sentiment analysis** as an alternative to numeric ratings
- **Causal inference methods** (propensity score matching, instrumental variables) to move beyond correlation

---
