# Used Car Price Prediction Using Machine Learning
**B.Tech Final Year Machine Learning Capstone Project & Interactive Web Application**

A full-stack, enterprise-grade Machine Learning web application designed for predicting used-car valuations, analyzing vehicle market distributions, discovering unsupervised clusters, modelling sequential maintenance states with Hidden Markov Models (HMM), and benchmarking classical, regularized, and ensemble regression models.

---

## 🏎️ Key Highlights & Architecture
- **Automotive-Themed Executive Dashboard**: Modern dark navy (`#0B192C`), slate (`#1E3E62`), and electric blue (`#008DDA`) interface with responsive KPI cards, Plotly charts, and real-time filters.
- **Dynamic Dataset Engine**: Automatically inspects, cleans, imputes missing values, and one-hot encodes any uploaded used-car dataset without manual column hardcoding.
- **Academic Rigor (Units 1 to 5)**:
  - **Unit 1**: Descriptive statistics, Bayes' Theorem simulation, quantiles, Pearson/Spearman correlations, polynomial curve fitting.
  - **Unit 2**: Linear, Polynomial (degree 2/3), Ridge (L2), SVR (RBF kernel), and Logistic Regression with confusion matrix and ROC-AUC.
  - **Unit 3**: K-Means, Gaussian Mixture Models (GMM), Hierarchical Agglomerative clustering (Dendrogram), PCA (2D & 3D projections), Elbow method, and Silhouette analysis.
  - **Unit 4**: Discrete Markov Chains, stationary distributions, and complete Hidden Markov Model (HMM) inference (Forward, Backward, and Viterbi algorithms) on dedicated sequential telemetry.
  - **Unit 5**: CART Decision Trees, Random Forest, Gradient Boosting (GBDT), AdaBoost, Stacking Ensemble, MDI Feature Importance, and Bayesian Model Averaging (BMA).
- **Dedicated Valuation Portal**: Dynamic form inputs matching the dataset schema, 95% confidence intervals, and automated SQLite prediction logging with CSV export.

---

## 📁 Project Folder Structure

```
used_car_price_prediction/
├── app.py                             # Main Streamlit web application & router
├── requirements.txt                   # Production package dependencies
├── README.md                          # Full setup, architecture & viva guide
├── .streamlit/
│   └── config.toml                    # Streamlit custom theme configuration
├── data/
│   ├── used_cars_sample.csv           # Reference used car dataset (1,500 records)
│   ├── sequential_maintenance_sample.csv # Sequential vehicle telemetry for HMM
│   ├── used_cars_app.db               # SQLite database (auto-generated on launch)
│   └── generate_datasets.py           # Synthetic dataset generator script
├── modules/
│   ├── __init__.py
│   ├── database.py                    # SQLite persistence for prediction history
│   ├── data_loader.py                 # Dynamic CSV loader, cleaner & preprocessor
│   ├── module1_dashboard.py           # Module 1: Executive Dashboard & KPIs
│   ├── module2_data_analysis.py       # Module 2: Unit 1 Statistics & Bayes' Rule
│   ├── module3_regression_classification.py # Module 3: Unit 2 Supervised ML
│   ├── module4_clustering_pca.py      # Module 4: Unit 3 Unsupervised Clustering
│   ├── module5_sequential_models.py   # Module 5: Unit 4 Markov & HMM Algorithms
│   ├── module6_ensemble_learning.py   # Module 6: Unit 5 Tree Ensembles & BMA
│   └── module7_price_prediction.py    # Module 7: Used Car Valuation Calculator
├── utils/
│   ├── __init__.py
│   ├── styling.py                     # Custom CSS styles, cards & layout helpers
│   └── hmm_algorithms.py              # Pure NumPy Forward, Backward & Viterbi
└── tests/
    └── test_modules.py                # Automated unit and integration test suite
```

---

## 🛠️ Technology Stack
| Layer | Technologies Used |
|---|---|
| **Programming Language** | Python 3.9+ (compatible with macOS standard Python 3.9.6) |
| **Web Framework / UI** | Streamlit (v1.35+) |
| **Machine Learning** | Scikit-learn (v1.3+) |
| **Data Processing** | Pandas, NumPy |
| **Scientific Computing** | SciPy |
| **Interactive Visualizations** | Plotly Express & Plotly Graph Objects |
| **Database** | SQLite3 (built-in standard library) |

---

## 💻 Step-by-Step VS Code Setup Instructions for macOS

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Press `Cmd + O` (or go to `File` > `Open Folder...`).
3. Navigate to:
   ```bash
   /Users/mohammedsahilts/.gemini/antigravity/scratch/used_car_price_prediction
   ```
4. Click **Open**.

### Step 2: Open the Integrated Terminal in VS Code
1. Press `` Ctrl + ` `` (backtick) or go to `Terminal` > `New Terminal` in the top menu.
2. Confirm you are in the project folder:
   ```bash
   pwd
   # Expected output: .../used_car_price_prediction
   ```

### Step 3: Set Up the Python Virtual Environment
Run the following commands in the VS Code terminal:

```bash
# 1. Create a dedicated Python 3 virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip (optional)
pip install --upgrade pip

# 4. Install all required dependencies
pip install -r requirements.txt
```

### Step 4: Configure Python Interpreter in VS Code
1. In VS Code, press `Cmd + Shift + P` to open the Command Palette.
2. Type `Python: Select Interpreter` and press `Enter`.
3. Choose the interpreter located at:
   `./.venv/bin/python` (Python 3.9.x).

---

## 🚀 Launching the Application

Make sure your virtual environment is active (`source .venv/bin/activate`), then run:

```bash
streamlit run app.py
```

Streamlit will launch local web servers and automatically open the application in your browser at:
```
http://localhost:8501
```

To stop the application at any time, press `Ctrl + C` in the terminal.

---

## 🧪 Running the Automated Test Suite

To verify that all algorithms, data pipelines, HMM decoders, and regression models are functioning with 100% mathematical accuracy:

```bash
source .venv/bin/activate
python tests/test_modules.py
```

Expected output:
```
✓ test_data_loading_and_cleaning passed
✓ test_modeling_data_bundle passed
✓ test_database_operations passed
✓ test_hmm_algorithms passed
✓ test_regression_and_classification_models passed
✓ test_clustering_and_pca passed
✓ test_ensemble_learning_models passed

🎉 ALL TESTS PASSED SUCCESSFULLY!
```

---

## 📂 Instructions for Uploading Your Own Dataset

1. In the application sidebar on the left, locate the **"📂 Dataset Management"** section.
2. Click **"Browse files"** or drag and drop your `.csv` file.
3. The application will instantly:
   - Identify your actual column names and data types.
   - Detect the target price column (e.g. `price`, `selling_price`, `cost`, `value`).
   - Clean non-numeric characters (currency symbols `$`, `₹`, comma formatting, units like `mi`, `km`).
   - Perform median imputation for numeric missing values and mode imputation for categorical missing values.
   - Dynamically re-train and update all 7 modules, charts, and prediction forms.
4. If no file is uploaded, the app seamlessly defaults to the verified built-in dataset (`data/used_cars_sample.csv`, 1,500 records), ensuring **zero blank pages or errors**.

---

## 🎓 College Project Review & Viva Defense Guide

Use this section to confidently demonstrate each module during your B.Tech review:

### 1. Module 1: Executive Dashboard
- **What to show**: Top KPI cards (Total Records, Features, Average/Min/Max Price), interactive price distribution with boxplot, top brands by inventory, mileage depreciation curve, and model year trajectory.
- **Viva Question**: *Why is median price reported alongside mean price?*
- **Defense Answer**: *Car price distributions are typically right-skewed with long tails of luxury vehicles. While the sample mean is sensitive to extreme outliers, the median ($P_{50}$) provides a robust measure of central tendency.*

### 2. Module 2: Data Analysis (Unit 1)
- **What to show**:
  1. **Descriptive Statistics**: Mean, median, mode, variance, standard deviation, skewness, kurtosis.
  2. **Bayes' Theorem Simulation**: Select a condition (e.g. `Brand = BMW`) and observe how the prior probability of being a premium car ($P(H)$) updates to the posterior probability ($P(H \mid E)$) via:
     $$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$$
  3. **Quantiles & Outlier Diagnostics**: ECDF curve and outlier detection comparing the IQR rule ($[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) with Z-Score rule ($|Z| > 3$).
  4. **Polynomial Curve Fitting**: Degrees 1 (linear), 2 (quadratic), and 3 (cubic) fits of Price vs Mileage/Year, displaying mathematical equations and $R^2$.

### 3. Module 3: Regression & Classification (Unit 2)
- **What to show**:
  1. Train **Linear Regression (OLS)**, **Polynomial Regression**, **Ridge Regression** ($L_2$ penalty $\alpha \sum w_i^2$), and **Support Vector Regression** (SVR with RBF kernel).
  2. Explain evaluation metrics: MAE, MSE, RMSE, and $R^2$.
  3. Inspect Actual vs Predicted plots and residual distributions.
  4. **Logistic Regression Tab**: Discretizes prices into Budget vs Premium tiers. Displays Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC curve.

### 4. Module 4: Clustering & PCA (Unit 3)
- **What to show**:
  1. **K-Means Clustering**: Segmenting vehicles into market clusters with centroid profile table.
  2. **Gaussian Mixture Models (GMM)**: Soft probabilistic clustering displaying BIC and AIC metrics.
  3. **Hierarchical Agglomerative Clustering**: Dendrogram visualization showing distance thresholds.
  4. **PCA**: Scree plot showing explained variance ratios, followed by 2D and 3D PCA subspace projections.
  5. **Diagnostics**: The Elbow Method (WCSS curve) and Silhouette Score curve identifying optimal $k$.

### 5. Module 5: Sequential Data Modelling (Unit 4)
- **Critical Theoretical Defense**: *Why are cross-sectional used car listings not sequential data?*
- **Defense Answer**: *Tabular car sales are independent, cross-sectional observations (i.i.d.). They do not represent a single system evolving over time. Imposing an HMM on unordered listings violates the Markov Property ($P(S_t \mid S_{t-1}, \dots) = P(S_t \mid S_{t-1})$). Therefore, we provide a dedicated sequential telemetry suite representing vehicle degradation states over maintenance steps.*
- **What to show**:
  1. Discrete Markov Chain Transition Matrix $A$ and stationary distribution $\pi_{\infty}$.
  2. Hidden Markov Model parameters $\lambda = (A, B, \pi)$.
  3. **Forward Algorithm**: Solves the Evaluation Problem ($\alpha_t(i)$ trellis matrix).
  4. **Backward Algorithm**: Backward recursion ($\beta_t(i)$ trellis matrix).
  5. **Viterbi Algorithm**: Solves the Decoding Problem to find the most likely hidden health state path $Q^*$.
  6. **Interactive Sequence Tester**: Enter any symptom sequence to see real-time decoding and state posteriors.

### 6. Module 6: Advanced & Ensemble Learning (Unit 5)
- **What to show**:
  1. Train **CART Decision Tree**, **Random Forest (Bagging)**, **Gradient Boosting (GBDT)**, **AdaBoost**, and **Stacking Regressor**.
  2. Benchmark Leaderboard comparing Test $R^2$, MAE, and RMSE.
  3. **Feature Importance**: Gini / Mean Decrease in Impurity (MDI) identifying top price drivers (e.g., Year, Mileage, Brand).
  4. **Bayesian Model Averaging (BMA)**: Demonstrates how posterior model probabilities are computed from BIC ($\Delta \text{BIC}$) to weight predictions across multiple models under hypothesis uncertainty.

### 7. Module 7: Dedicated Price Predictor
- **What to show**:
  1. Select Brand, Model, Year, Mileage, Fuel, Transmission, Accident status.
  2. Select model (e.g. Random Forest or Bayesian Blend).
  3. Click **"Predict Fair Market Price"**.
  4. Inspect predicted valuation, 95% Confidence Interval, and comparison against brand/year averages.
  5. Demonstrate SQLite persistence in the history table below and download the history as a CSV file.
