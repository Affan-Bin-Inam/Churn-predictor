# Customer Churn Predictor

A machine learning system that predicts whether a telecom customer is likely to churn, served through a FastAPI backend and a Streamlit frontend. This is a **learning/portfolio project** — it demonstrates a complete, working ML workflow from raw data to a deployed prediction interface, not a production-grade system.

---

## 1. Project Overview

Customer churn — a customer canceling their subscription — is one of the most direct threats to revenue for a subscription-based business like a telecom provider. Identifying customers who are likely to churn *before* they leave allows a retention team to intervene (e.g., with a discount or outreach), which is generally far cheaper than acquiring a new customer to replace a lost one.

This project predicts, from a customer's account and service attributes, whether they are likely to churn. It was built as a first end-to-end deployable ML project, intentionally scoped to reinforce classical ML fundamentals (classification, evaluation, tuning) while introducing the additional skills needed to ship a model as a working application: serialization, an API, input validation, and a frontend.

**This is a portfolio/learning project, not a production system.** Section 20 (Limitations) is explicit about where this implementation falls short of production standards.

---

## 2. Project Objectives

This project was built to learn and demonstrate:

- Data preprocessing on a real, messy tabular dataset
- Exploratory understanding of a mixed categorical/numeric dataset
- Feature engineering (bucketing, count features)
- Binary classification with Logistic Regression and Random Forest
- Evaluation with precision, recall, F1-score, and confusion matrices
- Reasoning about class imbalance and choosing an evaluation metric aligned with a business objective
- Cross-validation as a more reliable alternative to a single train/test split
- Hyperparameter tuning with `GridSearchCV`, explicitly optimized for recall
- Model serialization with `joblib`
- Building a REST API with FastAPI
- Input validation and data transformation with Pydantic (including computed fields)
- Building a simple frontend with Streamlit
- Connecting a frontend to a backend inference API
- The practical gap between "a model that scores well" and "a model wired into a usable system"

---

## 3. Dataset

**Source:** Telco Customer Churn dataset (public dataset, widely used for churn-prediction tutorials and benchmarks). *`<ADD EXACT SOURCE URL/PLATFORM HERE — e.g. Kaggle link — if you want it credited in the README>`*

**Size:** 7,043 customer records.

**Target variable:** `Churn` — whether the customer left the company (`Yes`/`No`).

**Class balance:** approximately 73.5% did not churn, 26.5% churned — a real, moderate class imbalance that shaped the modeling and evaluation choices throughout this project (see Section 9).

**Key raw attributes used:**

| Category | Attributes |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 4. Data Preprocessing

The following preprocessing was performed on the raw dataset before training:

- **`TotalCharges` type correction.** In the raw data, `TotalCharges` was not a clean numeric column — a small number of rows (corresponding to customers with `tenure = 0`, i.e. brand-new customers with no billing history yet) contained blank/invalid values instead of a number. These were converted to numeric (`pd.to_numeric(..., errors='coerce')`), and the resulting missing values were filled with the column median.
- **Categorical encoding.** All categorical columns (`gender`, `Contract`, `PaymentMethod`, the service columns, etc.) were one-hot encoded using `pandas.get_dummies(..., drop_first=True)`, dropping one category per column as the baseline to avoid redundant, perfectly-collinear columns.
- **Train/test split.** The data was split 80/20 into training and test sets, with the resulting test-set class balance (see Section 6 and 8 results) consistent with a stratified split preserving the ~73.5/26.5 class ratio.

**Important architectural note:** the preprocessing logic (missing-value handling, encoding) is **not** stored inside `churn_model.pkl`. The `.pkl` file contains only the trained Logistic Regression model's learned parameters. All preprocessing is re-implemented separately, in code, inside the FastAPI backend, and must be kept manually synchronized with what was done during training. This is explained further, as a real limitation, in Section 12.

---

## 5. Feature Engineering

Two engineered features made it into the final model's feature set:

- **`TenureCategory`** — the raw `tenure` (in months) was bucketed into five categories:

  | Tenure (months) | Category |
  |---|---|
  | ≤ 5 | New |
  | 6–11 | Developing |
  | 12–23 | Established |
  | 24–47 | Loyal |
  | ≥ 48 | Veteran* |

  *In the actual codebase, this category is spelled `"Vetran"` (a typo introduced early and kept intentionally for consistency between training and inference, rather than risking a mismatch by fixing it in only one place).

  `New` was used as the dropped baseline category during one-hot encoding, so the final feature set contains `TenureCategory_Developing`, `TenureCategory_Established`, `TenureCategory_Loyal`, and `TenureCategory_Vetran`.

- **`TotalServices`** — a count of how many of the following seven Yes/No service columns equal `"Yes"` for a given customer: `OnlineSecurity`, `OnlineBackup`, `PhoneService`, `StreamingMovies`, `StreamingTV`, `TechSupport`, `DeviceProtection`.

**A third engineered feature, `avg_monthly_spend` (`TotalCharges` divided by `tenure`, with a fallback to `MonthlyCharges` for zero-tenure customers to avoid division by zero), was explored during development but was not included in the final model's feature set** — testing showed it did not meaningfully improve results over the two features above, and it was dropped in favor of using `TotalCharges` directly.

---

## 6. Model Development

Two models were trained and compared throughout this project: **Logistic Regression** and **Random Forest Classifier**. Comparing a simple linear model against a non-linear ensemble model is a useful default check — it reveals whether the relationships in the data are closer to additive/linear or driven by non-linear feature interactions, which shapes which model is worth investing further tuning effort into.

**Initial baseline results** (default hyperparameters, no class weighting, test set of 1,409 customers):

**Logistic Regression:**
```
[[934 102]
 [149 224]]
```
| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (No churn) | 0.86 | 0.90 | 0.88 |
| 1 (Churn) | 0.69 | 0.60 | 0.64 |

Accuracy: 0.82

**Random Forest:**
```
[[938 98]
 [202 171]]
```
| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (No churn) | 0.82 | 0.91 | 0.86 |
| 1 (Churn) | 0.64 | 0.46 | 0.53 |

Accuracy: 0.79

**Logistic Regression outperformed Random Forest from this very first comparison**, particularly on recall and F1 for the churn class — a pattern that held throughout later iterations of this project (see Section 23).

---

## 7. Cross-Validation

Rather than trusting a single train/test split, 5-fold cross-validation (`cross_val_score`, `cv=5`) was used during hyperparameter exploration. A single split can be misleadingly lucky or unlucky depending on which rows happen to land in the test set; cross-validation trains and evaluates the model 5 times on different folds and reports the mean and standard deviation, giving a more reliable estimate of how the model is likely to perform on new data.

---

## 8. Hyperparameter Tuning

**Random Forest** was first explored manually, one parameter at a time, before any automated search:
- `n_estimators` (10, 50, 100, 200) — performance plateaued after ~50 trees, with diminishing returns beyond that.
- `max_depth` (2, 4, 6, None) — traced out a clear bias-variance pattern: `max_depth=2` underfit, `max_depth=4–6` performed best, and unlimited depth showed early signs of overfitting.

Both models were then tuned with `GridSearchCV`:

**Random Forest:**
```python
GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [4, 6, 8, None], 'min_samples_split': [2, 5, 10]},
    cv=5, scoring='recall', n_jobs=-1
)
```
Best parameters: `{'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 100}`, best cross-validated recall: **0.8155**

**Logistic Regression:**
```python
GridSearchCV(
    estimator=LogisticRegression(max_iter=10000, class_weight='balanced'),
    param_grid={'C': [0.01, 0.1, 1, 10, 100]},
    cv=5, scoring='recall', n_jobs=-1
)
```
Best parameter: `{'C': 100}`, best cross-validated recall: **0.7948**

**What these settings control:**
- **`C`** is the inverse of regularization strength for Logistic Regression — a smaller `C` applies stronger regularization (simpler decision boundary, more resistant to overfitting), while a larger `C` (like the selected `100`) allows the model to fit the training data more closely.
- **`class_weight='balanced'`** was used for both models to counteract the ~73.5/26.5 class imbalance — it penalizes misclassifications of the minority (churn) class more heavily during training, pushing the model to catch more true churners at the cost of more false alarms.
- **`scoring='recall'`** was deliberately chosen over the default (accuracy) because, per the project's own framing (Section 9), missing an actual churner was judged more costly than mistakenly flagging a loyal customer.

---

## 9. Precision vs. Recall — The Business Tradeoff

- **Recall** answers: *of the customers who actually churned, how many did the model catch?*
- **Precision** answers: *of the customers the model predicted would churn, how many actually did?*

For this project, **recall was prioritized over precision**. The reasoning: missing a customer who was going to churn (a false negative) means losing that customer entirely, whereas incorrectly flagging a loyal customer (a false positive) mainly costs an unnecessary retention offer — a real cost, but a smaller one than losing the customer outright.

This is a genuine tradeoff, not a free win: optimizing for recall means accepting more false positives. In the final tuned Logistic Regression model, this tradeoff is visible directly in the confusion matrix in Section 23 — a meaningful number of loyal customers are flagged as at-risk in exchange for catching more real churners.

---

## 10. `predict()` vs. `predict_proba()`

- `model.predict()` returns the final predicted class: `0` (no churn) or `1` (churn), based on the default 0.5 probability threshold.
- `model.predict_proba()` returns an array of two probabilities per prediction: `[probability_of_class_0, probability_of_class_1]`.

The FastAPI backend returns **both** the class prediction and the class-1 (churn) probability, so a consumer of the API can see not just the verdict but the model's confidence in it.

The 0.5 decision threshold used here is scikit-learn's default and was not tuned in this project. **Threshold tuning is a natural future improvement** — since recall was explicitly prioritized, deliberately lowering the classification threshold below 0.5 could push recall even higher (at a further cost to precision) without retraining the model at all.

---

## 11. Model Serialization

```python
joblib.dump(best_log_model, 'churn_model.pkl')

model_columns = data.drop('Churn', axis=1).columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')
```

Two separate files are saved:
- **`churn_model.pkl`** — the trained, tuned Logistic Regression model itself.
- **`model_columns.pkl`** — the exact list (and order) of the 34 feature column names the model was trained on, *after* one-hot encoding.

`model_columns.pkl` exists to solve a specific problem: a single new customer's raw input, once one-hot encoded, will only ever produce *some* of the trained columns (e.g., if their `Contract` is `"Month-to-month"`, no `Contract_One year` or `Contract_Two year` column is produced at all). At inference time, the incoming data is `reindex`-ed against `model_columns`, and any column the new input didn't produce is filled in as `False` (equivalent to `0` in a boolean one-hot column). **`model_columns.pkl` does not contain any preprocessing logic** — it is purely a list of expected column names, used to align a new input's shape to what the model expects.

---

## 12. Important Architectural Constraint: Preprocessing Is Not in the Model File

This is a significant, deliberate simplification worth stating plainly: **the saved `.pkl` file contains only the trained model, not a full preprocessing pipeline.** As a result, the FastAPI backend must manually reproduce the exact same preprocessing and encoding steps used during training — the same `get_dummies` column list, the same category naming, the same computed-feature logic.

This is fragile by construction: if the training notebook and the API's preprocessing code ever drift out of sync (e.g., a category is renamed or a feature's calculation changes on one side but not the other), the model will silently receive incorrectly-shaped input and produce wrong predictions with no error raised. This did happen during development (see Section 20).

**A stronger, production-appropriate implementation would use a single scikit-learn `Pipeline`** (bundling a `ColumnTransformer` for encoding together with the model) and persist that entire pipeline as one artifact — removing the need to hand-maintain matching preprocessing code in two places. This is listed as a future improvement in Section 22.

---

## 13. Case Sensitivity / Yes-No Normalization

The training data represents categorical values with a specific, fixed casing (e.g., `"Yes"`/`"No"`), but a real API caller might reasonably send `"yes"`, `"YES"`, or other variations. Pydantic `field_validator`s in the `CustomerInput` model normalize incoming text fields (lowercasing and re-mapping to the exact casing the trained model expects) before the data reaches the encoding step. This avoids the need to retrain the model just to accommodate input casing variation, but it does mean the API's accepted values must be kept in sync with whatever exact category strings exist in the training data — a mismatch here (e.g., an unexpected category value) would silently produce an all-zero one-hot row for that field rather than a visible error, unless explicitly validated against.

---

## 14. Pydantic Validation

FastAPI receives and validates incoming requests through a `CustomerInput` Pydantic model, which is used for:

- **Type validation** (e.g., `Tenure` must be an integer, `MonthlyCharges` a float)
- **Required fields** — every field needed by the model must be present in the request
- **`Literal` constraints** where appropriate (e.g., `Contract` restricted to `"Month-to-month"`, `"One year"`, `"Two year"`)
- **Input normalization** via `field_validator`s (case normalization, and converting `SeniorCitizen`'s `"Yes"`/`"No"` input into the `0`/`1` integer the model expects)
- **Computed fields**, calculated automatically from other input fields:
  - `TenureCategory` — derived from `Tenure`, per the bucketing rules in Section 5
  - `TotalServices` — derived by counting the seven relevant Yes/No service fields
  - `TotalCharges` — derived as `MonthlyCharges × Tenure`. **This is an approximation**, not the true historical billing figure used during training (which came from real accumulated billing data); for a genuinely new customer, no real historical `TotalCharges` value exists yet, so this estimate is used as a stand-in.

---

## 15. FastAPI Backend

The backend (`main.py`) loads `churn_model.pkl` and `model_columns.pkl` once, at application startup — not on every request, since neither file changes while the server is running. On each request to the prediction endpoint, it:

1. Validates the incoming JSON against `CustomerInput` (including running the computed fields).
2. Converts the validated input into a single-row DataFrame.
3. One-hot encodes the categorical fields, using the same column list used during training.
4. Reindexes the resulting row against `model_columns`, filling any columns the single input didn't produce with `False`.
5. Runs `model.predict()` and `model.predict_proba()` on the aligned row.
6. Returns a JSON response with the prediction and probability.

**Endpoint:**

```
POST /predict
```

**Example request body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 65.50
}
```

**Example response:**
```json
{
  "churn_prediction": false,
  "churn_probablility": 0.34
}
```

*(Note: the response key is spelled `churn_probablility` in the current implementation — a naming typo carried through from early development. It's left as-is here to accurately document the current API contract; fixing it is a trivial but real breaking change for anything already consuming this endpoint, noted in Section 22.)*

---

## 16. Streamlit Frontend

A separate Streamlit application (`frontend.py`) provides a simple, human-usable form for entering a customer's attributes (grouped into "Customer Profile" and "Services" sections, using `st.form` so the page doesn't rerun on every widget interaction). On submission, it sends the collected data as a JSON request to the FastAPI `/predict` endpoint and displays:

- The churn prediction (styled as a success or error message depending on the verdict)
- The churn probability, shown both as a percentage and a progress bar

This creates a simple three-layer separation: **Frontend (Streamlit) → API (FastAPI) → ML Model (Logistic Regression)** — the frontend has no knowledge of how the model works, it only knows how to call the API and display the result.

---

## 17. Project Architecture

```
User
 ↓
Streamlit Frontend (frontend.py)
 ↓
FastAPI /predict endpoint (main.py)
 ↓
Pydantic validation (CustomerInput)
 ↓
Computed fields (TenureCategory, TotalServices, TotalCharges)
 ↓
Categorical encoding (pd.get_dummies)
 ↓
Column alignment (reindex against model_columns.pkl)
 ↓
Logistic Regression model (churn_model.pkl)
 ↓
Prediction + probability (JSON response)
 ↓
Streamlit displays result to user
```

---

## 18. Project Structure

**Files that actually exist in this project:**
```
churn-predictor/
├── main.py              # FastAPI backend
├── frontend.py           # Streamlit frontend
├── churn_model.pkl        # Trained, tuned Logistic Regression model
├── model_columns.pkl      # Expected feature column list/order
├── requirements.txt       # Pinned dependencies
└── README.md
```

**Recommended additions (not currently part of the repo):**
```
├── data/                  # Raw/processed dataset (if included in the repo)
└── notebooks/             # Training notebook(s) with EDA, model comparison, and tuning
```

---

## 19. How to Run (Windows)

```bash
# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint directly through FastAPI's interactive Swagger UI.

In a separate terminal (with the same virtual environment activated):
```bash
streamlit run frontend.py
```

This opens the Streamlit form in your browser (typically `http://localhost:8501`), which calls the running FastAPI server automatically. **The FastAPI server must be running for the Streamlit frontend to return predictions.**

---

## 20. Limitations

This is a first, small-scale learning project, and the following limitations are genuine, not hedging:

- **Manual preprocessing rather than a persisted pipeline** — as detailed in Section 12, the encoding logic is duplicated between training and inference code rather than saved as a single artifact.
- **`model_columns.pkl` preserves feature structure only, not preprocessing logic.**
- **`TotalCharges` at inference time is an approximation** (`MonthlyCharges × Tenure`), not a true historical billing value, since a new customer has no real billing history yet.
- **No automated data validation beyond Pydantic's request-level checks** — there is no check that a submitted combination of features is internally realistic.
- **The model is unreliable on internally-inconsistent inputs.** During validation, submitting a DSL customer with a monthly charge far below the observed range for DSL (below the training data's minimum, and closer to the "no internet service" price range) produced an erratic, high-confidence prediction. This suggests the model has implicitly learned joint relationships between features (e.g., typical pricing per service tier) and behaves unpredictably outside them — a known limitation of tabular models without explicit input consistency checks.
- **Predictions for brand-new customers (`tenure = 0`) are low-confidence in practice**, even though the model doesn't express this itself. Only 11 such customers exist in the training data, and all 11 did not churn — too few examples for the model to have reliably learned this segment's behavior.
- **No monitoring, drift detection, authentication, database, or automated retraining.**
- **Not yet deployed to a public host** — the application currently runs locally only (FastAPI via `uvicorn`, Streamlit locally); no live URL exists yet.
- **The classification threshold is scikit-learn's untuned default (0.5)** — not adjusted specifically for this project's stated recall priority.
- **A known cosmetic bug**: the API response key is misspelled (`churn_probablility` instead of `churn_probability`), left uncorrected in the current version to accurately document the live API contract.
- **Churn probability is a model estimate**, not a guaranteed real-world probability — it reflects patterns in historical data, not certainty about any individual customer's future behavior.

---

## 21. What I Learned

- A trained model is only one component of a working ML system — the preprocessing, the interface, and the serving logic all matter as much as the model itself.
- Preprocessing must match *exactly* between training and inference; a subtle mismatch (a leftover, now-obsolete workaround was one real example encountered during this project) can silently corrupt predictions with no error raised.
- Feature engineering doesn't always outperform simpler fixes — in this project, addressing class imbalance with `class_weight='balanced'` moved the model's recall far more than the engineered features did.
- The right evaluation metric depends on the business problem, not a default choice — recall was deliberately prioritized here because missing a churner was judged costlier than a false alarm, and that decision shaped every subsequent tuning choice.
- Cross-validation gives a materially more trustworthy performance estimate than a single train/test split.
- Hyperparameter tuning should optimize for a metric that's actually meaningful for the problem (`scoring='recall'` here), not accuracy by default.
- Saving a model is not the same as saving a complete inference pipeline — this gap was the source of the most persistent bugs in this project.
- Probing a model with deliberately crafted, single-variable-changed inputs is a genuinely useful way to sanity-check what it has learned, and to discover where it becomes unreliable.
- Turning a notebook into an API and a frontend is a distinct skill set from training the model, and surfaces failure modes (type mismatches, serialization format mismatches, encoding drift) that never appear inside a notebook.

---

## 22. Future Improvements

- Combine preprocessing and the model into a single scikit-learn `Pipeline` (using `ColumnTransformer` for categorical/numerical handling), and persist that pipeline as one artifact instead of maintaining matching logic in two places.
- Add input consistency validation (e.g., checking that submitted pricing is plausible for the selected service tier) to catch and flag out-of-distribution requests before trusting the model's output.
- Experiment with tuning the classification threshold instead of relying on the default 0.5.
- Calibrate predicted probabilities if probability quality becomes important for downstream decisions.
- Conduct a more thorough hyperparameter search (e.g., a wider grid, or `RandomizedSearchCV`/Bayesian search for efficiency).
- Try gradient boosting models (XGBoost, LightGBM) for comparison against Logistic Regression and Random Forest.
- Further feature engineering, informed by feature importance/coefficient inspection.
- Add automated tests for the preprocessing and prediction logic.
- Add logging and basic monitoring.
- Fix the `churn_probablility` response key typo (a breaking change for any existing consumer, so worth doing deliberately with versioning in mind).
- Containerize the application with Docker.
- Deploy the FastAPI backend and Streamlit frontend to a public host.

---

## 23. Results

**Baseline (default hyperparameters, no class weighting):**

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.69 | 0.60 | 0.64 | 0.82 |
| Random Forest | 0.64 | 0.46 | 0.53 | 0.79 |

**With `class_weight='balanced'` and engineered features (`TenureCategory`, `TotalServices`):**

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.53 | 0.82 | 0.64 | 0.76 |
| Random Forest | 0.66 | 0.47 | 0.55 | 0.79 |

**Final, `GridSearchCV`-tuned models (both optimized for recall):**

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy | Best CV Recall |
|---|---|---|---|---|---|
| Logistic Regression (`C=100`) | **0.52** | 0.82 | **0.64** | **0.76** | 0.7948 |
| Random Forest (tuned) | 0.49 | **0.85** | 0.62 | 0.73 | 0.8155 |

**Selected model: Logistic Regression.** The two tuned models are close, with Random Forest catching marginally more churners (85% vs. 82% recall). Logistic Regression was chosen because it wins on precision, F1, and accuracy, is substantially simpler and faster, and — importantly for a churn-intervention use case — is more directly interpretable (its coefficients can be inspected to explain *why* a customer was flagged, which is genuinely useful to a retention team deciding how to act on a flagged case).

---

## 24. Portfolio Positioning

This project is not "I trained a model and it got a good score." It demonstrates a complete, working pipeline:

**data → preprocessing → feature engineering → evaluation → model selection → serialization → API → frontend → inference**

Specifically, it shows the ability to reason about a real class-imbalance problem and choose an evaluation metric aligned with a stated business objective, rather than defaulting to accuracy; to diagnose an unexpected model result methodically rather than accepting it at face value; to identify and fix real integration bugs between a trained model and its serving code; and to communicate a model's limitations honestly rather than overstating its readiness for production. It is intentionally scoped as a learning project, and is documented as such.
