# 🏀 WNBA Game Outcome Prediction

A full end-to-end data science project analyzing WNBA game logs (2015–2025) to identify the key performance statistics that predict whether a team wins or loses.

---

## Project Overview

The WNBA receives far less analytical attention than the NBA, despite being a professional league with rich statistical data. This project addresses that gap by building machine learning models to predict game outcomes and uncover what truly drives winning in the WNBA.

**Core Hypothesis:** Higher shooting efficiency — measured by field goal percentage (FG%) and three-point percentage (3P%) — is associated with winning games.

---

## Dataset

- **Source:** [WNBA Game Logs 2015–2025 on Kaggle](https://www.kaggle.com/datasets/natoshakennebrew/wnba-gamelogs-2015-2024)
- **File:** `wnba_gamelogs_2015_2025.csv`
- **Coverage:** Regular season game logs for all WNBA teams from 2015 to 2025
- **Size:** 4,513 rows × 48 columns
- **Each row** represents a single team's box score performance in one game

The dataset includes team and opponent statistics (points, field goals, rebounds, steals, blocks, turnovers, fouls, shooting percentages) plus advanced metrics like offensive/defensive ratings, possession estimates, and turnover rates.

---

## Project Structure

The notebook is organized across 5 sequential project phases:

### Project 1 — Problem Definition & Study Design
- Defined the research question and hypothesis
- Identified independent variables (shooting stats, rebounding, turnovers, etc.) and the dependent variable (Win/Loss)
- Outlined potential confounders: player injuries, roster changes, and officiating inconsistency
- Developed a data collection plan emphasizing multi-season coverage to avoid team-level bias

### Project 2 — Data Cleaning & Validation
- Loaded and profiled the dataset (shape, dtypes, unique values, descriptive statistics)
- Renamed columns for clarity: `G#` → `Game_Number`, `W/L` → `Win_Loss`
- Verified there were no duplicate rows or missing values
- Identified and corrected a data error: `Opp_tov_rate` was incorrectly set to 1 for all rows — recalculated using the proper possession formula
- Verified all derived shooting percentages (FG%, 3P%, FT%) against their raw components for both team and opponent
- Converted the `Date` column to proper datetime format
- Detected outliers using Z-scores across all 40+ numeric columns and visualized the top 10 with boxplots
- Verified extreme values (e.g., 14 blocks in a single game) against official Basketball Reference box scores — retained as genuine performances
- Exported a cleaned dataset for downstream use

### Project 3 — Exploratory Data Analysis
- Re-examined the cleaned dataset: confirmed 4,513 rows with zero duplicates or missing values
- Converted `Win_Loss` from W/L strings to binary (1/0)
- Confirmed shooting percentage columns are stored as proportions (0–1), not raw percentages
- Categorized all variables by type: categorical, date, and numeric
- Analyzed each variable's distribution, summary statistics, and outlier count using IQR
- Explored categorical variables (Season, Team, Home/Away, Opponent, Win/Loss) with count plots — noted the 2020 shortened COVID season (264 games vs. ~400+ in other years)
- Visualized the distribution of team points: most games scored between 80–90 points
- Explored periodicity: plotted the New York Liberty's average points per season (2015–2025), showing a COVID dip in 2020 and strong offensive resurgence from 2021 onward
- Built a correlation heatmap across key metrics — notable findings:
  - `Tm_Pts` and `Tm_off_rating`: +0.89
  - `Tm_AST` and `Tm_Pts`: +0.61
  - `Tm_def_rating` and `Win_Loss`: −0.56
- Plotted FG% vs. Points Scored colored by Win/Loss — confirmed that efficient shooting strongly separates wins from losses
- Tracked the league-wide rise in 3-pointers made per game from 2015–2025 (under 5 → over 8), showing a strategic shift toward perimeter offense
- Demonstrated **Simpson's Paradox** in WNBA shooting data: identified season-level team pairs (e.g., SEA vs. MIN) where one team had lower 2P% and 3P% individually but a higher overall FG% due to differing shot-type distributions

### Project 4 — Feature Engineering, Dimensionality Reduction & Baseline Modeling
- Framed the ML task as a **supervised binary classification** problem (Win vs. Loss)
- Removed identifiers, raw counting stats, and leakage-prone outcome variables (e.g., `Tm_Pts`, `win_margin`)
- Standardized all features using `StandardScaler` (zero mean, unit variance)
- Applied **PCA** to reduce 27 features to 17 principal components while retaining 95% of total variance — a 37% dimensionality reduction
- Implemented two additional feature selection strategies as extra credit:
  - **Filter method** (ANOVA F-test via `SelectKBest`) — selected top 15 features based on statistical correlation with Win/Loss
  - **Wrapper method** (forward selection with KNN via `SequentialFeatureSelector`) — selected top 15 features based on model performance
- Trained a **Decision Tree Classifier** as a baseline model using PCA-reduced features
- Tuned hyperparameters (`max_depth`, `min_samples_leaf`, `min_samples_split`) via **GridSearchCV** with 5-fold cross-validation
- Best parameters found: `max_depth=4`, `min_samples_leaf=10`, `min_samples_split=2`
- Decision Tree on PCA features achieved **~95.5% accuracy** but reduced interpretability due to abstract PCA components

### Project 5 — Refined Feature Selection & Model Comparison
- Shifted away from PCA toward domain-informed, interpretable feature selection
- Selected 17 fundamental basketball statistics (shooting efficiency, rebounding, turnovers, fouls, steals, blocks for both teams) — excluding advanced derived metrics to prevent data leakage
- Applied **RFECV** (Recursive Feature Elimination with Cross-Validation) with three different base estimators to identify the optimal feature subset for each model
- Trained and evaluated four models:

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 0.941 | 0.942 | High accuracy; signs of overfitting |
| L2-Regularized Logistic Regression | 0.941 | ~0.942 | Coefficient shrinkage did not remove features |
| Decision Tree (depth=5) | — | ~0.80 | Lower accuracy; some misclassification |
| Random Forest (100 trees, depth=5) | 0.942 | 0.858 | Best balance of accuracy and generalizability |

- Evaluated all models using accuracy, precision, recall, F1-score, classification report, and confusion matrix heatmaps
- Concluded that **Random Forest** offered the best trade-off: reduced overfitting vs. the logistic models and stronger performance vs. a single decision tree

---

## Key Findings

- **The hypothesis is supported:** Higher FG% and 3P% are among the strongest predictors of winning. RFECV consistently selected shooting efficiency metrics as critical features across models.
- **Defense matters as much as offense.** Opponent turnover rate, opponent rebounding, and defensive metrics were consistently retained by feature selection — winning is not purely an offensive story.
- **The 3-point shot has become central to WNBA strategy.** Three-pointers made per game nearly doubled from 2015 to 2025.
- **Simpson's Paradox is present in WNBA shooting data.** A team can shoot better in every individual shot category yet show a lower overall FG% — a cautionary note for aggregate stat analysis.
- **Data leakage is a real concern in sports datasets.** Variables like points scored and win margin encode the outcome and must be excluded from features.

---

## Technologies Used

- **Python** (Google Colab)
- **pandas** — data loading, cleaning, transformation
- **NumPy** — numerical operations
- **matplotlib / seaborn** — visualization
- **scikit-learn** — preprocessing, feature selection (RFECV, SelectKBest), PCA, model training and evaluation
- **mlxtend** — sequential forward feature selection
- **scipy** — Z-score outlier detection

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/natoshakennebrew/wnba-gamelogs-2015-2024) and upload it to your Google Drive as `wnba_gamelogs_2015_2025.csv`
2. Open `wnba.ipynb` in Google Colab
3. Mount your Google Drive when prompted
4. Run all cells in order — each project phase builds on the previous one

---

