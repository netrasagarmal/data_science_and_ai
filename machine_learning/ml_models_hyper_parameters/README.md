# ML Model Hyperparameters, Selection Scenarios & Evaluation Guide

> Context: Churn prediction pipeline with 6 model families — logistic regression, shallow decision tree, LightGBM, XGBoost, calibrated LightGBM deep, and small neural net.

---

## Table of Contents

1. [Logistic Regression](#1-logistic-regression)
2. [Shallow Decision Tree](#2-shallow-decision-tree)
3. [LightGBM](#3-lightgbm)
4. [XGBoost](#4-xgboost)
5. [Calibrated LightGBM Deep](#5-calibrated-lightgbm-deep)
6. [Small Neural Net](#6-small-neural-net)
7. [Model Selection Scenarios](#7-model-selection-scenarios)
8. [Comparing Models During Evaluation](#8-comparing-models-during-evaluation)

---

## 1. Logistic Regression

### What it is
A linear model that estimates the probability of a binary outcome using a sigmoid function. It draws a linear decision boundary in feature space.

### Key Hyperparameters

| Hyperparameter | Common Values | Importance | Why it matters |
|---|---|---|---|
| `C` (inverse regularisation strength) | `0.001, 0.01, 0.1, 1, 10, 100` | 🔴 Critical | Controls overfitting. Low C = heavy regularisation (simpler model, high bias). High C = weak regularisation (complex model, high variance). Most important parameter to tune. |
| `penalty` | `l1`, `l2`, `elasticnet`, `none` | 🔴 Critical | L1 drives coefficients to zero (automatic feature selection). L2 shrinks all coefficients. ElasticNet is a mix. For high-dimensional feature tables (144 features), L1 or ElasticNet often works better. |
| `solver` | `lbfgs`, `liblinear`, `saga` | 🟡 Medium | `liblinear` is good for small datasets. `saga` is required for L1 + large datasets. `lbfgs` is the default for L2. Must be compatible with chosen penalty. |
| `max_iter` | `100, 500, 1000` | 🟡 Medium | Number of optimisation iterations. If model hasn't converged, increase this. Watch for convergence warnings in training logs. |
| `class_weight` | `None`, `balanced` | 🔴 Critical | Set to `balanced` for imbalanced churn datasets. Automatically weights minority class (churners) higher so the model doesn't ignore them. |
| `fit_intercept` | `True`, `False` | 🟢 Low | Almost always `True`. Set to `False` only if data is already centred. |

### When NOT to use
- Non-linear relationships between features and churn (most real-world churn data)
- When interactions between features matter (e.g. high usage + high complaints = higher churn risk)

---

## 2. Shallow Decision Tree

### What it is
A single decision tree with restricted depth. Makes splits based on feature thresholds to partition data into homogeneous groups.

### Key Hyperparameters

| Hyperparameter | Common Values | Importance | Why it matters |
|---|---|---|---|
| `max_depth` | `2, 3, 4, 5` | 🔴 Critical | The most important parameter. Shallow depth (2-4) forces interpretability and prevents overfitting. Deep trees memorise training data. For a cascade ensemble entry point, depth 2-3 is typical. |
| `min_samples_split` | `2, 10, 50, 100` | 🔴 Critical | Minimum samples required to split a node. Higher values prevent overfitting on small leaf nodes. Especially important with imbalanced churn data. |
| `min_samples_leaf` | `1, 5, 10, 20` | 🔴 Critical | Minimum samples in a leaf node. Controls granularity. Increasing this smooths the model and prevents tiny, unreliable leaves. |
| `criterion` | `gini`, `entropy` | 🟡 Medium | `gini` is faster and usually performs similarly to `entropy`. `entropy` can sometimes find better splits on imbalanced datasets. |
| `max_features` | `None`, `sqrt`, `log2`, float | 🟡 Medium | Number of features considered at each split. Reducing this introduces randomness and can help in ensemble settings. |
| `class_weight` | `None`, `balanced` | 🔴 Critical | Same as logistic regression — set `balanced` for churn imbalance. |
| `splitter` | `best`, `random` | 🟢 Low | `best` finds optimal splits. `random` introduces randomness (useful when used as base estimator in ensemble). |

### When NOT to use
- As your primary production model for probability outputs (poorly calibrated probabilities)
- When you need high accuracy on ambiguous cases (use in cascade as early rejector instead)

---

## 3. LightGBM

### What it is
A gradient boosting framework using histogram-based learning and leaf-wise tree growth. Faster and more memory-efficient than XGBoost for large datasets. 

### Key Hyperparameters

| Hyperparameter | Common Values | Importance | Why it matters |
|---|---|---|---|
| `n_estimators` / `num_boost_round` | `100–2000` | 🔴 Critical | Number of boosting rounds (trees). Too few = underfitting, too many = overfitting. Always pair with early stopping. |
| `learning_rate` / `eta` | `0.01, 0.05, 0.1, 0.3` | 🔴 Critical | Step size for each boosting round. Lower learning rate + more trees = better generalisation but slower training. Classic trade-off: lower LR needs higher n_estimators. |
| `max_depth` | `-1, 4, 6, 8` | 🔴 Critical | Maximum tree depth. `-1` means no limit (leaf-wise growth handles this). LightGBM grows leaf-wise, so `num_leaves` is more important than `max_depth`. |
| `num_leaves` | `15, 31, 63, 127` | 🔴 Critical | **Most important LightGBM-specific parameter.** Controls model complexity. Should be `< 2^max_depth`. More leaves = more complex model. Start with 31, tune carefully. |
| `min_child_samples` / `min_data_in_leaf` | `20, 50, 100, 200` | 🔴 Critical | Minimum samples in a leaf. Critical for preventing overfitting on rare churners. Increase for imbalanced datasets. |
| `subsample` / `bagging_fraction` | `0.6–1.0` | 🟡 Medium | Fraction of training data used per tree. Introducing randomness reduces overfitting and speeds up training. |
| `colsample_bytree` / `feature_fraction` | `0.6–1.0` | 🟡 Medium | Fraction of features used per tree. Reduces correlation between trees, improves ensemble diversity. |
| `reg_alpha` (L1) | `0, 0.1, 1, 10` | 🟡 Medium | L1 regularisation on leaf weights. Drives sparse solutions. Helps when many features are irrelevant. |
| `reg_lambda` (L2) | `0, 0.1, 1, 10` | 🟡 Medium | L2 regularisation on leaf weights. Smooths predictions. Usually the first regularisation param to tune. |
| `scale_pos_weight` | `ratio of negatives to positives` | 🔴 Critical | For imbalanced churn data. Set to `(total non-churners) / (total churners)`. Equivalent to `class_weight='balanced'`. |
| `early_stopping_rounds` | `50, 100` | 🔴 Critical | Stops training when validation metric stops improving. Prevents overfitting and saves compute. Always use in training. |
| `boosting_type` | `gbdt`, `dart`, `goss` | 🟡 Medium | `gbdt` is standard. `dart` uses dropout (slower but sometimes better). `goss` is faster by sampling gradients. |

---

## 4. XGBoost

### What it is
A gradient boosting framework using level-wise tree growth. More regularisation options than LightGBM. Generally more robust out-of-the-box but slower on large datasets.

### Key Hyperparameters

| Hyperparameter | Common Values | Importance | Why it matters |
|---|---|---|---|
| `n_estimators` | `100–2000` | 🔴 Critical | Same as LightGBM. Number of trees. Always use with early stopping. |
| `learning_rate` / `eta` | `0.01, 0.05, 0.1, 0.3` | 🔴 Critical | Same trade-off as LightGBM. |
| `max_depth` | `3, 4, 5, 6, 8` | 🔴 Critical | Maximum depth of each tree. Unlike LightGBM (leaf-wise), XGBoost grows level-wise so `max_depth` directly controls complexity. Default 6 is often a good start. |
| `min_child_weight` | `1, 3, 5, 10` | 🔴 Critical | Minimum sum of instance weights in a child. Higher values prevent splits on small/noisy groups. Analogous to `min_child_samples` in LightGBM. Key overfitting control. |
| `gamma` / `min_split_loss` | `0, 0.1, 0.5, 1` | 🟡 Medium | Minimum loss reduction required to make a split. Acts as pruning. `0` means any split is allowed. Increase to make the model more conservative. |
| `subsample` | `0.6–1.0` | 🟡 Medium | Row sampling per tree. Same function as LightGBM `bagging_fraction`. |
| `colsample_bytree` | `0.6–1.0` | 🟡 Medium | Feature sampling per tree. |
| `colsample_bylevel` | `0.6–1.0` | 🟢 Low | Feature sampling per tree level. More granular than `colsample_bytree`. Use for fine-tuning after other params. |
| `reg_alpha` (L1) | `0, 0.1, 1` | 🟡 Medium | L1 regularisation. Encourages sparsity in leaf weights. |
| `reg_lambda` (L2) | `1, 5, 10` | 🟡 Medium | L2 regularisation. Default is 1 (XGBoost regularises by default, unlike some libraries). |
| `scale_pos_weight` | `negatives / positives` | 🔴 Critical | Same as LightGBM — critical for churn imbalance. |
| `tree_method` | `hist`, `gpu_hist`, `exact` | 🟡 Medium | `hist` is fast and approximate (similar to LightGBM). `exact` is slower but precise. On Databricks with GPU, use `gpu_hist`. |
| `max_delta_step` | `0, 1` | 🟢 Low | Helps with convergence on highly imbalanced datasets. Setting to 1 can improve stability when `scale_pos_weight` is very large. |

### LightGBM vs XGBoost — Key Differences

| Aspect | LightGBM | XGBoost |
|---|---|---|
| Tree growth | Leaf-wise (faster, can overfit) | Level-wise (more stable) |
| Speed on large data | Faster | Slower |
| Memory usage | Lower | Higher |
| Key complexity param | `num_leaves` | `max_depth` |
| Default regularisation | Less (need to tune more) | More (reg_lambda=1 default) |
| Categorical features | Native support | Requires encoding |

---

## 5. Calibrated LightGBM Deep

### What it is
A deep (high-capacity) LightGBM model with post-hoc probability calibration applied (typically isotonic regression or Platt scaling). The "deep" refers to allowing higher model complexity — more leaves, more trees. Calibration corrects the raw probability outputs to be reliable.

### Key Hyperparameters

This model has **two layers of hyperparameters** to tune.

#### Layer 1 — LightGBM base model (deep configuration)

| Hyperparameter | Typical Deep Values | Importance | Why it matters |
|---|---|---|---|
| `num_leaves` | `63, 127, 255` | 🔴 Critical | Higher than standard LightGBM. More leaves capture complex non-linear churn patterns. Risk of overfitting — needs more regularisation to compensate. |
| `n_estimators` | `500–3000` | 🔴 Critical | More trees than shallow config. Pairs with lower learning rate. |
| `learning_rate` | `0.01, 0.02, 0.05` | 🔴 Critical | Lower than standard to accommodate more trees. |
| `min_child_samples` | `20–100` | 🔴 Critical | Must be carefully tuned in deep model — more complex splits mean more risk of tiny leaves. |
| `reg_lambda` | `1–20` | 🔴 Critical | **Increase regularisation** compared to standard LightGBM to compensate for higher complexity (more leaves). |
| `reg_alpha` | `0.1–5` | 🟡 Medium | Add L1 to drive unnecessary leaf weights to zero. |
| `colsample_bytree` | `0.5–0.8` | 🟡 Medium | Reduce feature fraction more aggressively for deep models to add regularisation through diversity. |
| `subsample` | `0.6–0.8` | 🟡 Medium | Row sampling also adds regularisation. |

#### Layer 2 — Calibration method

| Hyperparameter | Options | Importance | Why it matters |
|---|---|---|---|
| `method` (CalibratedClassifierCV) | `isotonic`, `sigmoid` | 🔴 Critical | `isotonic` is non-parametric — learns a monotone mapping. Better for large calibration sets (>1000 samples). `sigmoid` (Platt scaling) is parametric and less prone to overfitting on small sets. |
| `cv` | `prefit`, `3`, `5` | 🔴 Critical | `prefit` uses a held-out calibration set (recommended — avoids data leakage). Cross-validated calibration can leak if the same data trained the model. **Always use a separate calibration split in a churn pipeline.** |

#### Why calibration matters here specifically
Deep LightGBM models tend to push probabilities to extremes (very close to 0 or 1). This means a raw probability of 0.7 may not mean "70% of customers with this score actually churn." Calibration fixes this — after calibration, the probabilities are reliable enough to use as direct business inputs (e.g. "intervene if churn probability > 0.6").

---

## 6. Small Neural Net

### What it is
A fully connected feed-forward neural network with 1–3 hidden layers. For tabular churn data, "small" typically means 2 hidden layers with 64–256 neurons each.

### Key Hyperparameters

| Hyperparameter | Common Values | Importance | Why it matters |
|---|---|---|---|
| `hidden_layer_sizes` | `(64,), (128, 64), (256, 128, 64)` | 🔴 Critical | Architecture of the network. Too small = underfitting. Too large = overfitting on tabular data. For 144 input features, start with 1-2 hidden layers. |
| `learning_rate` | `0.001, 0.0001, 0.01` | 🔴 Critical | Step size for gradient descent. Too high = unstable training. Too low = slow convergence. Adam optimizer is less sensitive to this than SGD. |
| `batch_size` | `32, 64, 128, 256` | 🟡 Medium | Samples per gradient update. Smaller batches = noisier updates but often better generalisation. Larger batches = faster but can converge to sharp minima. |
| `epochs` / `max_iter` | `50–300` | 🔴 Critical | Training iterations. Use early stopping with a validation set — don't set a fixed epoch count. |
| `dropout_rate` | `0.1–0.5` | 🔴 Critical | Fraction of neurons randomly deactivated during training. Primary regularisation for neural nets on tabular data. Prevents co-adaptation of neurons. Typically 0.2–0.3 for churn. |
| `optimizer` | `adam`, `sgd`, `rmsprop` | 🟡 Medium | Adam is the default choice for most tabular tasks. Adaptive learning rate per parameter. SGD with momentum can generalise better if tuned carefully. |
| `activation` | `relu`, `elu`, `leaky_relu` | 🟡 Medium | Non-linearity of hidden layers. `relu` is standard. `elu` can help with dying neuron problem. Avoid sigmoid/tanh in hidden layers (vanishing gradients). |
| `weight_decay` / `l2_reg` | `1e-4, 1e-3, 1e-2` | 🟡 Medium | L2 regularisation on weights. Penalises large weights. Analogous to `reg_lambda` in tree models. |
| `batch_normalisation` | `True`, `False` | 🟡 Medium | Normalises layer inputs. Stabilises training and can allow higher learning rates. Often helps on tabular data with mixed feature scales. |
| `early_stopping patience` | `10, 20, 30` | 🔴 Critical | Stops training when validation loss stops improving for N epochs. Prevents overfitting. |
| `class_weight` | `None`, `{0:1, 1:w}` | 🔴 Critical | Weight the minority class (churners) higher in the loss function. Essential for imbalanced churn data. |

---

## 7. Model Selection Scenarios

### When to use each model

| Scenario | Recommended Model | Reason |
|---|---|---|
| Stakeholder needs to explain every prediction | Logistic Regression or Shallow Decision Tree | Coefficients / tree paths are human-readable |
| Regulatory audit requires transparent decision logic | Shallow Decision Tree | Can be printed and reviewed by non-technical auditors |
| Fast inference required at scale (millions of customers daily) | Logistic Regression or Shallow Decision Tree | Microsecond inference, no tree traversal overhead |
| Dataset is small (< 5,000 labelled samples) | Logistic Regression | Less prone to overfitting; tree models need more data to find reliable splits |
| Many irrelevant features, want automatic selection | Logistic Regression (L1) or LightGBM | L1 drives weights to zero; LightGBM's feature importance identifies useful features |
| Large dataset (millions of rows, 100+ features) | LightGBM | Fastest training via histogram binning; handles large feature tables natively |
| Dataset has many categorical features | LightGBM | Native categorical handling; XGBoost requires manual encoding |
| Business needs reliable probability scores for prioritisation | Calibrated LightGBM Deep | Deep model captures complex patterns + calibration ensures probabilities are trustworthy |
| Need to set a probability threshold for budget-constrained intervention | Calibrated LightGBM Deep | Only calibrated models give reliable thresholds (e.g. "intervene above 0.6") |
| Strong non-linear interactions expected (e.g. usage + complaints) | LightGBM, XGBoost, or Neural Net | Capture feature interactions that linear models miss |
| High-quality, curated dataset with careful feature engineering | XGBoost | More conservative growth (level-wise) rewards good feature engineering |
| Embedding-like representations needed for downstream clustering | Small Neural Net | Hidden layer activations can serve as learned embeddings |
| First model in a Viola-Jones cascade (quick rejection of easy negatives) | Shallow Decision Tree or Logistic Regression | Fast, interpretable, low latency; handles obvious cases well |
| Final model in a cascade (handles hard ambiguous cases) | Calibrated LightGBM Deep or XGBoost | High capacity for difficult borderline predictions |
| Ensemble member needing maximum diversity from tree models | Small Neural Net | Completely different inductive bias from gradient boosting |

### When NOT to select each model

| Model | Avoid when |
|---|---|
| Logistic Regression | Data has strong non-linear patterns or feature interactions |
| Shallow Decision Tree | High accuracy is required (limited capacity by design) |
| LightGBM | Dataset is very small — leaf-wise growth overfits aggressively |
| XGBoost | Dataset is very large and training speed is a constraint |
| Calibrated LightGBM Deep | Fast model iteration / experimentation needed (heavy to train and calibrate) |
| Small Neural Net | Dataset is small (< 10,000 rows), or interpretability is required |

---

## 8. Comparing Models During Evaluation

### Step 1 — Establish evaluation metrics before training

Define your primary and secondary metrics upfront. Changing them after seeing results introduces cherry-picking.

| Metric | Formula / Description | When to use as primary |
|---|---|---|
| **AUC-ROC** | Area under the ROC curve. Threshold-independent. | When ranking customers by risk (not setting a fixed cut-off) |
| **AUC-PR** | Area under Precision-Recall curve | Preferred for imbalanced datasets where AUC-ROC can be misleadingly high |
| **F1 Score** | Harmonic mean of precision and recall | When you care equally about both false positives and false negatives |
| **Precision @ K** | Precision in the top-K highest-scored customers | When you have a fixed intervention budget (top 10,000 customers) |
| **KS Statistic** | Max separation between TPR and FPR curves | Common in telecom/banking for model quality reporting |
| **Log Loss / Brier Score** | Measures probability calibration quality | Required when using raw probabilities for business decisions |
| **Lift / Gain curve** | How much better the model is vs. random at each decile | Communicating business value to non-technical stakeholders |

### Step 2 — Ensure a fair comparison

```
All models must be evaluated on:
  - The same held-out test set (never seen during training or hyperparameter tuning)
  - The same temporal split (train on older data, test on more recent data)
  - The same stratified class distribution in train/validation/test splits
  - The same feature set (no model should have access to features others don't)
```

### Step 3 — Evaluation comparison table structure

Build a comparison table with these columns:

| Model | AUC-ROC | AUC-PR | F1 | Precision@10K | Brier Score | Inference Time (ms) | Training Time (min) | Notes |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | | | Baseline |
| Shallow Decision Tree | | | | | | | | Cascade entry |
| LightGBM | | | | | | | | |
| XGBoost | | | | | | | | |
| Calibrated LightGBM Deep | | | | | | | | Calibrated probs |
| Small Neural Net | | | | | | | | |
| **Ensemble** | | | | | | | | **Champion candidate** |

### Step 4 — Statistical significance testing

A small difference in AUC (e.g. 0.87 vs 0.872) may be noise, not a real improvement. Use these tests:

| Test | Purpose | When to use |
|---|---|---|
| **DeLong test** | Compare two AUC-ROC scores for statistical significance | Comparing any two models on the same test set |
| **McNemar's test** | Compare two classifiers' error patterns | When you want to know if errors are on the same customers |
| **Bootstrap confidence intervals** | Build confidence intervals around any metric | Always — report metric ± CI, not just point estimate |

```python
# Example: Bootstrap confidence interval for AUC
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import numpy as np

aucs = []
for _ in range(1000):
    idx = resample(range(len(y_test)))
    aucs.append(roc_auc_score(y_test[idx], y_pred[idx]))

print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"95% CI: [{np.percentile(aucs, 2.5):.4f}, {np.percentile(aucs, 97.5):.4f}]")
```

### Step 5 — Beyond accuracy: production-readiness criteria

When two models have similar accuracy, decide using:

| Criterion | Description | Weight in decision |
|---|---|---|
| **Calibration quality** | Reliability diagram / calibration curve — is predicted probability = actual probability? | High — needed for threshold-based actions |
| **Inference latency** | Time to score one customer. Critical if scoring millions daily. | High for real-time, low for batch |
| **Feature importance stability** | Do feature importances change significantly across different train/test splits? | Medium — unstable importances = unreliable model |
| **Performance on minority class** | Recall and precision specifically on churners (the positive class) | High — a model with high overall AUC can still miss most churners |
| **Degradation over time** | Score model on progressively more recent test windows. Does AUC drop? | High — models that degrade fast are expensive to maintain |
| **Interpretability** | Can a business stakeholder understand why this customer was flagged? | High if regulatory/audit requirements exist |
| **Training reproducibility** | Given the same data and hyperparameters, do you get the same model? | Medium — important for debugging and audits |

### Step 6 — Champion/Challenger decision framework

```
Champion selection criteria (in order of priority):

1. Statistical tie-breaking:
   - If AUC difference is NOT statistically significant → compare Brier Score
   - If Brier Score is similar → compare Precision@K (budget-constrained use case)

2. If metrics are still tied:
   - Prefer simpler model (logistic regression > shallow tree > LightGBM)
   - Reason: lower maintenance burden, faster inference, easier to explain

3. Override conditions where complex model wins despite tie:
   - Business stakeholders confirmed they need calibrated probabilities
   - Downstream personalisation/cohort system depends on probability quality
   - AUC-PR (not just AUC-ROC) shows meaningful difference on minority class

4. Document the decision:
   - Record winning model, runner-up, metrics for both, and decision rationale
   - Register in MLflow/Unity Catalog with all evaluation artefacts attached
```

### Step 7 — Visualisations to produce for every model comparison

| Plot | What it shows | Tools |
|---|---|---|
| ROC curve (all models on one plot) | AUC-ROC comparison | `sklearn.metrics.RocCurveDisplay` |
| Precision-Recall curve | AUC-PR comparison, especially useful for imbalanced data | `sklearn.metrics.PrecisionRecallDisplay` |
| Calibration curve (reliability diagram) | How well probabilities match actual rates | `sklearn.calibration.CalibrationDisplay` |
| Confusion matrix (at chosen threshold) | FP/FN breakdown per model | `sklearn.metrics.ConfusionMatrixDisplay` |
| Lift/Gain chart | Business value at each decile | Custom or `scikit-plot` |
| Feature importance / SHAP beeswarm | What each model is learning | `shap`, `lightgbm.plot_importance` |
| Learning curve | Bias/variance diagnosis | `sklearn.model_selection.learning_curve` |

---

## Quick Reference — Hyperparameter Tuning Priority

| Model | Tune first | Tune second | Tune last |
|---|---|---|---|
| Logistic Regression | `C`, `penalty` | `class_weight`, `solver` | `max_iter` |
| Shallow Decision Tree | `max_depth`, `min_samples_leaf` | `class_weight`, `criterion` | `max_features` |
| LightGBM | `num_leaves`, `learning_rate`, `n_estimators` | `min_child_samples`, `reg_lambda` | `subsample`, `colsample_bytree` |
| XGBoost | `max_depth`, `learning_rate`, `n_estimators` | `min_child_weight`, `gamma` | `subsample`, `colsample_bytree`, `reg_alpha` |
| Calibrated LightGBM Deep | Same as LightGBM, then calibration `method` and `cv` | `reg_lambda` (increase for deep) | `colsample_bytree`, `subsample` |
| Small Neural Net | `hidden_layer_sizes`, `learning_rate`, `dropout_rate` | `batch_size`, `weight_decay` | `activation`, `optimizer` |

