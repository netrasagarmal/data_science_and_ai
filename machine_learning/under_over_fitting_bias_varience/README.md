# Complete guide: Underfitting, overfitting, bias and variance across 9 ML models

We'll first build a common framework you can apply to any ML model, then go through all nine models you listed.

# 1. The fundamental concepts

## 1.1 What is underfitting vs. overfitting?

Imagine you're building a telecom churn prediction model. You have historical customer data, and you want to predict which customers will leave.

[The Ultimate Machine Learning Cheat Sheet for Non-Specialists | ML4Industry energy Blog | ML4Industry](https://images.openai.com/static-rsc-4/ddb7muQVDihWfGScH--e_Uw7EA8kjAQHhyms6MkgE7qkR_kzMNxFAhfSNMWpRkl4UZ7YBuAlaTt4qtw-Qtz0NzW9ZsTbZw1XCa1A5RnVYq-yaVABtEw8KIKR5sPiIQiuga8baG8WVcjO59Up_06RSKtIfGR6gQVL1GCO_OC3cvs?purpose=inline)

Underfitting — the model has not learned enough

The model is too simple, or its training is too restricted, to learn the actual patterns. It performs poorly on both training and unseen data.

Example: churn depends on tenure, network quality, billing issues and their interactions, but the model captures only a simple linear relationship.

[A Comprehensive Guide to Stratified K-Fold Cross-validation for Unbalanced Data](https://images.openai.com/static-rsc-4/PCRkepvb7PzoS4CH5KescVo15F073FiiecJZfsI3iffFhG670z9ju_bQdJAwvlk5fA6A6Yh1X9rZtrc12JAN6uNxCBC21Rznd7QSQDK9BPfEWlMh1ZJG35cRQlJR9iVvbH9bKk3JsFbzsOprZCQUBNYgeWcFSqqJ4DwVrLU-76c?purpose=inline)

Good fit — the model learns generalizable patterns

The model captures meaningful relationships in the training data and performs well on unseen customers.

[Example of overfitting and underfitting in machine learning](https://images.openai.com/static-rsc-4/FycFkdAYOmssPKJ1N8Y1QrxM4McllP_c-ny8mHhkyYrIF4bCr6OXn2qBEjLEiUPQ5H6rY3Esbg4eKjwujLBCvQlUigw8jeZirsfghYaFbrEGT49ib1ymOY5xed8IAguAUwJg59X0j9tLbL6OLI93jckEdiWVWEizkejGlf_Kit0?purpose=inline)

Overfitting — the model learns too much of the training data

It learns genuine patterns but also noise, outliers, accidental correlations or individual training examples. Training performance is excellent, but performance deteriorates on unseen data.

The key is that overfitting does not simply mean a complex model. It means the model has learned patterns that do not generalize to the population you care about.

## 1.2 How do you identify overfitting from training results?

Suppose you trained a binary classification model for churn prediction.

| Metric    | Training | Validation |
| --------- | -------- | ---------- |
| Accuracy  | 98%      | 78%        |
| Precision | 97%      | 75%        |
| Recall    | 96%      | 71%        |
| Log loss  | 0.05     | 0.52       |

The large performance gap is a warning sign. The model fits the training examples extremely well but struggles with unseen examples.

For regression, the equivalent might be:

| Metric | Training | Validation |
| ------ | -------- | ---------- |
| RMSE   | 2        | 15         |
| MAE    | 1.4      | 11         |

Lower RMSE and MAE are better. Here, the model's training errors are very low, but its validation errors are much higher.

There is no universal gap that automatically proves overfitting. Compare it with a baseline, the natural noise in the data, and results across validation folds.

### The four possible scenarios

| Training performance | Validation performance | Likely interpretation                                                         |
| -------------------- | ---------------------- | ----------------------------------------------------------------------------- |
| Poor                 | Poor                   | Underfitting, poor features, noisy data or a broken pipeline                  |
| Good                 | Poor                   | Overfitting, distribution mismatch or leakage-related evaluation problems     |
| Good                 | Good                   | Good generalization, assuming the test is representative                      |
| Poor                 | Good                   | Check split difficulty, metric variability, sample size and data distribution |

One important nuance: if training performance is poor but validation performance is much better, do not automatically call it underfitting. That pattern is unusual and warrants investigating the evaluation setup.

## 1.3 Bias–variance trade-off

$$Var(X) = E(X^2) - [E(X)]^2$$

For this normal curve with sigma 1.4, the identity above becomes

$$Var(X)=\sigma^2=(\text{1.4})^2=\text{1.96}$$

$$\sigma$$

μ-σ+σValue of XRelative density

Give feedback

Bias and variance explain two different sources of prediction error.

- High bias: the model's assumptions are too restrictive, so it consistently misses important relationships.
- High variance: the model is too sensitive to the particular training sample. Small changes in training data can produce substantially different models.

Underfitting → typically high bias

A linear model tries to represent a strongly nonlinear churn relationship with one straight line. Even with more training data, its restricted structure may continue to miss the pattern.

Overfitting → typically high variance

A very deep decision tree learns specific combinations of customer attributes and noise. A small change in the training sample can produce a different tree.

For squared-error regression, a useful conceptual decomposition is:

$$ \mathbb{E}[(Y-\hat f(X))^2] = \text{Bias}^2+\text{Variance}+\text{Irreducible noise} $$

This is a simplified decomposition under standard assumptions. For classification and losses such as log loss, the exact decomposition differs, but the intuition remains useful.

Important interview nuance: bias and variance are not the same thing as training error and validation error. The training–validation gap is an observable diagnostic; bias and variance describe why prediction error occurs.

## 1.4 What happens to training data when overfitting occurs?

This is a crucial distinction: overfitting does not necessarily change the dataset. It changes what the model learns from it.

Look for four categories of evidence.

1\. Insufficient or unrepresentative data

Too few examples, missing customer segments, limited examples of rare events, or a training population that differs from production.

2\. Noise, outliers and inconsistent labels

A small number of mislabeled churn outcomes or extreme billing values can lead flexible models to learn misleading patterns.

3\. Too many weak or redundant features

Hundreds of irrelevant features, high-cardinality identifiers, and accidental correlations give a flexible model more opportunities to fit noise.

4\. Leakage and an invalid split

Features that contain future information, duplicate customers across splits, or preprocessing performed before splitting can produce misleadingly good training or validation results.

You should not automatically remove outliers, delete features, or collect more data. First establish which issue is causing the generalization problem.

# 2. Before fixing a model: establish the correct experiment

Before looking at individual algorithms, use this workflow. It is especially important for enterprise datasets with many tables and millions of customer records.

1\. Split the data correctly

Train / validation / untouched test set

2\. Train a simple baseline

Compare against a dummy model or simple algorithm

3\. Compare train and validation metrics

Measure absolute performance and the generalization gap

4\. Diagnose the cause

Data quality, feature quality, model capacity, training or leakage

5\. Change one thing at a time

Tune, retrain and compare on the same validation strategy

### How should you split the data?

The correct split depends on the use case.

- Random split: suitable for independent, identically distributed observations.
- Stratified split: useful for classification when preserving class proportions matters, especially with rare churners.
- Time-based split: train on older data and validate on newer data when predicting future customer behavior.
- Group-based split: keep records belonging to the same customer, household, device or other relevant entity together when the deployment scenario requires generalization to unseen entities.

For a telecom churn model, a random split may be misleading if the same customer appears repeatedly or if features include information generated after the prediction date. Use a time-aware, customer-aware design appropriate to how the model will be deployed.

Fit imputers, scalers, encoders, feature selectors and other learned preprocessing steps on the training data only. Apply the fitted transformations to validation and test data.

# 3. Linear regression

Linear regression predicts a continuous numerical value, such as revenue, monthly customer spend or expected network usage.

Its main limitation is that it assumes the expected target can be represented by a linear combination of the chosen features, unless you explicitly add transformations or interactions.

## How to identify underfitting and overfitting

|                  | Underfitting                                               | Overfitting                                                           |
| ---------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- |
| Training error   | High                                                       | Very low                                                              |
| Validation error | High                                                       | Much higher than training                                             |
| Residuals        | Systematic patterns remain                                 | Training residuals may be tiny, but validation residuals are larger   |
| Typical cause    | Relationship is too complex for the feature representation | Too many features, unstable coefficients or noisy engineered features |

### Bias and variance

- Underfitting: high bias. A straight-line relationship cannot capture a strongly curved relationship.
- Overfitting: potentially high variance, especially when there are many correlated predictors, few observations or high-dimensional feature engineering.

Plain ordinary least squares with a fixed feature set does not automatically become highly flexible just because the sample is noisy. Overfitting risk often comes from the feature representation, feature selection, high dimensionality or repeated model selection.

### Dataset indicators

Possible underfitting causes

- Residuals systematically increase or decrease with a feature.
- The relationship between a predictor and target is nonlinear.
- Important interaction terms are missing.
- Important explanatory variables are unavailable.

Possible overfitting causes

- Number of predictors is large relative to the sample size.
- Many highly correlated or redundant features exist.
- There are outliers or erroneous labels.
- Features are selected repeatedly using the same validation set.

### Model and training indicators

Underfitting

- Linear model with no nonlinear transformations.
- Important interaction terms omitted.
- Relevant variables excluded.

Overfitting

- Large numbers of engineered features relative to available examples.
- High variance in coefficients across different training samples.
- Feature selection or polynomial expansion introduces too much flexibility.

### How to reduce each problem

To reduce underfitting

- Add meaningful nonlinear features, such as squared terms or splines.
- Add justified interaction terms.
- Include relevant domain features.
- Compare against more flexible models if the relationship remains nonlinear.

To reduce overfitting

- Apply Ridge (L2) or Lasso (L1) regularization.
- Reduce noisy or redundant features.
- Collect more representative observations where possible.
- Investigate influential outliers and data errors.
- Use cross-validation and nested validation when selecting features or hyperparameters.

Interview example: Suppose customer spend depends on tenure in a nonlinear way. If both train and validation errors are high, try a nonlinear feature representation. If training error is low but validation error is high, inspect feature dimensionality and use regularization rather than blindly adding more terms.

# 4. Logistic regression

Logistic regression predicts a probability for a classification target, such as whether a customer will churn.

It models the log-odds as a linear combination of features. The resulting probability is passed through a sigmoid function for binary classification.

## How to identify underfitting and overfitting

|                     | Underfitting      | Overfitting                          |
| ------------------- | ----------------- | ------------------------------------ |
| Training log loss   | High              | Low                                  |
| Validation log loss | High              | Much higher                          |
| ROC-AUC             | Poor on both sets | Stronger on training than validation |
| Precision/recall    | Poor on both      | Often drops on validation            |

For logistic regression, examine log loss and ROC-AUC alongside precision, recall and PR-AUC. Accuracy alone can hide poor performance on an imbalanced churn dataset.

### Bias and variance

- Underfitting: high bias because the linear decision boundary cannot represent the true relationship.
- Overfitting: high variance due to excessive or correlated predictors, sparse categories, weakly supported features or aggressive model selection.

### Dataset indicators

Underfitting

- Churn is influenced by nonlinear combinations of tenure, usage, billing and network quality.
- Important interactions are absent.
- Relevant signals are missing or poorly represented.

Overfitting

- Many weak features or rare one-hot categories.
- Sparse features with very few positive and negative examples.
- High-cardinality identifiers that act as proxies for individual customers.
- Label noise or leakage.

### Model and hyperparameter indicators

- Underfitting: excessive regularization, represented by a very large `C`? No — in scikit-learn, a small `C` means stronger regularization, so an excessively small `C` can cause underfitting.
- Overfitting: very weak regularization (large `C`), too many weak predictors, or repeated tuning on the same validation set.

### How to reduce each problem

Underfitting

- Add justified nonlinear transformations and interaction features.
- Reduce regularization by increasing `C` when appropriate.
- Improve feature representation and data quality.

Overfitting

- Increase regularization by decreasing `C`.
- Use Ridge-like L2 regularization or L1 regularization for sparsity.
- Remove leakage, IDs and unstable features.
- Combine rare categories where appropriate, based on training data and domain logic.
- Use stratified or time-based cross-validation, as appropriate.

Practical nuance: Logistic regression can overfit despite being a relatively simple model. It is not immune to overfitting merely because it has a linear decision boundary.

# 5. K-nearest neighbors (KNN)

KNN predicts using nearby training observations. For regression, it averages their target values; for classification, it uses their labels or weighted votes.

The main complexity control is the number of neighbors, $$k$$.

## How to identify underfitting and overfitting

|                  | Underfitting                         | Overfitting                                    |
| ---------------- | ------------------------------------ | ---------------------------------------------- |
| Training error   | Often high                           | Very low, particularly with $$k=1$$          |
| Validation error | High                                 | Higher than training                           |
| Model behavior   | Overly smooth, misses local patterns | Sensitive to individual observations and noise |

### Bias and variance

- Underfitting: high bias when $$k$$ is too large. Predictions average across observations that may belong to different patterns.
- Overfitting: high variance when $$k$$ is too small. Predictions depend heavily on individual neighbors.

### Dataset indicators

Underfitting

- Important local patterns are being averaged away.
- Distinct customer groups are treated as nearby despite meaningful differences.
- Relevant features are missing.

Overfitting

- Noisy observations or mislabeled neighbors.
- Irrelevant features distort distances.
- Features have incompatible scales.
- High-dimensional data makes neighborhoods less meaningful.

### Model and hyperparameter indicators

- Underfitting: very large `n_neighbors`, inappropriate distance metric, or excessive distance weighting toward broadly distributed points.
- Overfitting: `n_neighbors=1` or a very small $$k$$, an overly flexible distance configuration, or poor scaling.

### How to reduce each problem

Underfitting

- Decrease $$k$$ to retain more local structure.
- Try a more suitable distance metric.
- Add informative features.
- Test distance-weighted neighbors where justified.

Overfitting

- Increase $$k$$.
- Scale numerical features using a training-fitted scaler.
- Remove irrelevant or noisy features.
- Consider dimensionality reduction where appropriate.
- Tune $$k$$ using cross-validation.

Example: With $$k=1$$, a single mislabeled customer can determine a prediction. With $$k=1000$$, the prediction may be too averaged to identify a small but meaningful churn segment. The goal is to find a neighborhood size that generalizes.

# 6. Decision tree regression

A decision tree regressor repeatedly splits the feature space into regions and predicts a numerical value, typically the mean target value of training observations in each leaf.

Its flexibility is largely controlled by its depth, leaf size and split constraints.

## How to identify underfitting and overfitting

|                     | Underfitting                  | Overfitting                         |
| ------------------- | ----------------------------- | ----------------------------------- |
| Training error      | High                          | Extremely low, potentially zero     |
| Validation error    | High                          | Much higher than training           |
| Tree structure      | Too shallow or too restricted | Many deep branches and tiny leaves  |
| Prediction behavior | Misses important changes      | Learns small fluctuations and noise |

### Bias and variance

- Underfitting: high bias. A shallow tree cannot capture enough structure.
- Overfitting: high variance. A deep tree learns highly specific conditions that may change with another training sample.

### Dataset indicators

Underfitting

- Important nonlinear relationships and threshold effects are present but not captured.
- Important predictors are absent.
- Measurement noise is so high that useful structure is difficult to learn.

Overfitting

- Many outliers or noisy target values.
- Too few representative examples within specific feature combinations.
- Numerous rare categories or identifier-like features create highly specific splits.
- Training observations contain accidental patterns that do not persist in production.

### Model and hyperparameter indicators

| Hyperparameter          | Too restrictive → underfitting | Too flexible → overfitting |
| ----------------------- | ------------------------------ | -------------------------- |
| `max_depth`             | Too small                      | Too large or unrestricted  |
| `min_samples_leaf`      | Too large                      | Too small                  |
| `min_samples_split`     | Too large                      | Too small                  |
| `max_leaf_nodes`        | Too small                      | Too large                  |
| `min_impurity_decrease` | Too large                      | Too small                  |

### How to reduce each problem

- Underfitting: increase `max_depth`, decrease `min_samples_leaf`, or allow more leaves.
- Overfitting: limit `max_depth`, increase `min_samples_leaf`, increase `min_samples_split`, or prune the tree.
- Use cross-validation to choose the constraints.
- Investigate outliers and data errors, but remove observations only when justified.
- Add more representative training data when the learned splits are unstable due to insufficient examples.

### Important practical detail: pruning

Cost-complexity pruning balances fit against tree size. In scikit-learn, `ccp_alpha` controls pruning strength.

- Too much pruning → underfitting.
- Too little pruning → a potentially overfit tree.

A decision tree can have excellent training performance and poor validation performance even when its overall depth appears reasonable. Tiny leaves and unstable splits are also worth inspecting.

# 7. Decision tree classification

A classification tree is structurally similar to a regression tree, but it predicts classes or class probabilities. Its splits often optimize criteria such as Gini impurity or entropy.

## How to identify underfitting and overfitting

|                                 | Underfitting            | Overfitting                                                   |
| ------------------------------- | ----------------------- | ------------------------------------------------------------- |
| Training classification error   | High                    | Very low                                                      |
| Validation classification error | High                    | Much higher                                                   |
| Probability quality             | Often weak on both sets | May be excellent on train but poorly calibrated on validation |
| Tree structure                  | Too shallow             | Deep branches with very small leaves                          |

### Bias and variance

- Underfitting: high bias due to overly restrictive splits.
- Overfitting: high variance due to branches that memorize particular combinations of features and labels.

### Dataset indicators

Underfitting

- Churn depends on threshold effects and feature interactions the tree cannot represent with its current restrictions.
- Rare but important patterns are not captured.
- Features that distinguish the classes are missing.

Overfitting

- Label noise and mislabeled examples.
- Very rare combinations of categorical features.
- Many features with high cardinality.
- Very small positive-class counts within individual segments.

### Model and hyperparameter indicators

The main controls are the same as for regression trees: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes` and `ccp_alpha`.

The difference is that the tree optimizes classification criteria rather than regression error.

A particularly important warning sign is a tree with pure training leaves that contain very few examples. Such leaves can produce extreme probabilities even when those probabilities are unreliable.

### How to reduce each problem

Underfitting

- Allow more depth or more leaves.
- Reduce overly restrictive leaf-size and split constraints.
- Add informative predictors and relevant interaction signals.
- Evaluate whether the split criterion and feature representation are appropriate.

Overfitting

- Limit depth and prune.
- Increase minimum leaf size.
- Remove leakage and unstable identifiers.
- Use cross-validation.
- For imbalanced classes, evaluate PR-AUC, recall, precision and the business cost of errors rather than accuracy alone.

Do not assume that class weighting automatically solves overfitting. It changes the importance of class-specific errors, but the resulting model can still generalize poorly.

# 8. Random forest regression

A random forest combines many decision trees trained using bootstrap samples and randomized feature selection at splits. Its predictions are generally averaged across the trees.

The purpose of this combination is to reduce the variance of individual trees while retaining the ability to learn nonlinear relationships.

## How to identify overfitting in random forest

This directly addresses your original interview question.

Consider these results:

| Metric          | Training RMSE | Validation RMSE |
| --------------- | ------------- | --------------- |
| Random forest A | 3.0           | 4.0             |
| Random forest B | 1.0           | 12.0            |

Forest B has a much larger generalization gap. It fits training data exceptionally well, but its predictions deteriorate on unseen data.

This suggests overfitting, but you should also check whether the validation set differs from training or whether leakage, time drift or a split problem is responsible.

### Bias and variance in random forest

A single deep tree often has low bias but high variance. A random forest averages many such trees, which typically reduces variance.

However, a random forest does not eliminate overfitting by definition. If its trees are extremely flexible, its data is noisy, or its validation setup is flawed, it can still generalize poorly.

Increasing the number of trees usually stabilizes the ensemble rather than directly increasing its overfitting in the same way that increasing the depth of a single tree can. More trees reduce Monte Carlo variability in the ensemble prediction; they do not necessarily fix a generalization gap caused by overly flexible base trees.

### Dataset indicators

Underfitting

- Important predictors are missing or noisy.
- The underlying relationship cannot be captured adequately by the current feature representation.
- Tree constraints are too restrictive.

Overfitting

- Training data has noisy targets or extreme outliers.
- Too many weak predictors and high-cardinality identifiers.
- Sparse subgroups contain too few examples.
- The training population is not representative of future observations.

### Model and hyperparameter indicators

| Hyperparameter      | Underfitting risk                              | Overfitting risk                                                                                          |
| ------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `max_depth`         | Too small                                      | Unrestricted depth on noisy data                                                                          |
| `min_samples_leaf`  | Too large                                      | Too small                                                                                                 |
| `min_samples_split` | Too large                                      | Too small                                                                                                 |
| `max_features`      | Too few candidate features can limit learning  | More features at each split can increase tree similarity and sometimes reduce ensemble variance reduction |
| `n_estimators`      | Too few trees can produce unstable predictions | More trees alone are generally not the primary overfitting cause                                          |

The effect of `max_features` is not monotonic: allowing more features can help individual trees but also make them more correlated. The best setting depends on the data.

### How to reduce underfitting

- Increase depth or relax leaf constraints.
- Add meaningful predictors and improve feature representation.
- Check whether the training set is sufficiently informative.
- Increase the number of trees if ensemble predictions are still unstable, while recognizing that this does not necessarily reduce bias.

### How to reduce overfitting

- Increase `min_samples_leaf` and/or `min_samples_split`.
- Limit `max_depth`.
- Tune `max_features` to balance individual tree strength and tree diversity.
- Remove leakage and investigate noisy or unstable features.
- Collect more representative data if the model lacks examples for important subgroups.
- Use cross-validation to choose the settings.

## Why can random forest still overfit?

Imagine the true relationship between network quality and customer spending is fairly smooth. One tree may learn an accidental split at a particular signal-strength value because a handful of training customers have unusually high spending.

Other trees may learn different accidental thresholds. Averaging their predictions reduces some of that instability, but not all of it. If each tree has too little data per leaf, their individual predictions may remain noisy.

Increasing the minimum leaf size can stabilize these predictions.

Interview-ready answer: “I compare training and validation RMSE and inspect the generalization gap. If training error is low but validation error is high, I check for leakage, distribution shift and noisy features first. Then I tune `max_depth`, `min_samples_leaf`, `min_samples_split` and `max_features` with cross-validation. Increasing `n_estimators` can stabilize predictions, but it is not my primary fix for overfitting.”

# 9. Random forest classification

Random forest classification combines trees that vote on a class or aggregate class probabilities.

The same general principles as random forest regression apply, but classification requires attention to probability quality and class imbalance.

## How to identify underfitting and overfitting

|                        | Underfitting | Overfitting                         |
| ---------------------- | ------------ | ----------------------------------- |
| Training performance   | Weak         | Very strong                         |
| Validation performance | Weak         | Substantially weaker                |
| Log loss               | High on both | Much lower on train than validation |
| ROC-AUC / PR-AUC       | Weak on both | Often lower on validation           |

Accuracy can be misleading. If only 5% of customers churn, a model that predicts no churn for everyone achieves 95% accuracy but is not a useful churn model.

### Bias and variance

- Underfitting: high bias; the ensemble cannot capture the class boundaries because of overly restrictive trees or weak features.
- Overfitting: high variance; individual trees learn noisy class boundaries or tiny subgroups, and the ensemble does not fully correct them.

### Dataset indicators

Underfitting

- Important churn predictors are missing.
- Class boundaries depend on nonlinear interactions that the current tree constraints fail to capture.
- Data quality is poor.

Overfitting

- Label noise, leakage or highly specific identifiers.
- Rare categories and tiny subgroups.
- Very few positive examples in some training segments.
- Train and validation populations differ.

### Model and hyperparameter indicators

- Underfitting: excessively shallow trees, large minimum leaf sizes or too few candidate features.
- Overfitting: unrestricted depth, tiny leaves, weak validation discipline or unstable features.
- Class imbalance: the default decision threshold or class distribution can create poor recall or precision, even without overfitting.

### How to reduce each problem

Underfitting

- Relax tree constraints.
- Improve the feature set.
- Tune the number of candidate features and the split constraints.

Overfitting

- Increase leaf size and limit tree depth.
- Tune `max_features`.
- Use leakage-safe preprocessing and correct splits.
- Remove unstable features and improve data quality.
- Validate with PR-AUC, log loss, precision, recall and business-specific thresholds.

A threshold adjustment can improve business outcomes without changing the fitted model's underlying generalization. It is not, by itself, a cure for overfitting.

# 10. XGBoost regression

XGBoost is a gradient-boosted tree algorithm. Unlike random forest, which typically builds trees independently and averages their predictions, XGBoost builds trees sequentially, with each new tree improving the current ensemble's objective.

This sequential correction can learn complex relationships, but it can also fit noise if the ensemble becomes too flexible.

## How to identify underfitting and overfitting

|                                        | Underfitting | Overfitting                     |
| -------------------------------------- | ------------ | ------------------------------- |
| Training RMSE / MAE                    | High         | Very low                        |
| Validation RMSE / MAE                  | High         | Much higher                     |
| Training loss across boosting rounds   | Remains high | Continues decreasing            |
| Validation loss across boosting rounds | Remains high | Starts increasing after a point |

The most useful diagnostic is often the validation metric plotted against boosting rounds.

### Illustrative training pattern

Conceptual illustration — not measured data

Training lossValidation loss

025507510012050100150200

Lower loss is better. The validation minimum marks the illustrative best stopping point.

The training loss keeps improving, while validation loss bottoms out and then rises. This is a classic sign of overfitting during boosting.

### Bias and variance

- Underfitting: high bias when the trees are too weak, the learning rate is too restrictive for the available number of rounds, or regularization is too strong.
- Overfitting: high variance when the ensemble learns increasingly specific patterns that do not generalize.

### Dataset indicators

Underfitting

- Weak or missing predictive features.
- Strong nonlinearities or interactions not captured by the current feature representation.
- Poor target quality or highly noisy inputs.

Overfitting

- Noisy targets and outliers.
- High-dimensional, sparse or redundant features.
- Rare subgroups with insufficient observations.
- Leakage or training/validation distribution mismatch.

### Model and hyperparameter indicators

| Parameter          | Underfitting risk                        | Overfitting risk                                                         |
| ------------------ | ---------------------------------------- | ------------------------------------------------------------------------ |
| `max_depth`        | Too small                                | Too large                                                                |
| `min_child_weight` | Too large                                | Too small                                                                |
| `learning_rate`    | Too small with too few rounds            | Large values can fit too aggressively                                    |
| `n_estimators`     | Too few boosting rounds                  | Too many rounds without early stopping                                   |
| `reg_lambda` (L2)  | Excessive regularization                 | Too little regularization                                                |
| `reg_alpha` (L1)   | Excessive regularization                 | Too little regularization                                                |
| `subsample`        | Very small values can lose useful signal | Sampling less aggressively can increase tree similarity                  |
| `colsample_bytree` | Too few columns may limit learning       | Using all columns can sometimes increase complexity and tree correlation |

The impact of `subsample` and `colsample_bytree` depends on the data. They are stochastic regularization tools, not settings that should automatically be reduced to their minimum values.

### How to reduce underfitting

- Increase `max_depth` moderately.
- Reduce overly restrictive `min_child_weight`.
- Increase boosting rounds if training is still improving and the learning rate is small.
- Reduce excessive L1/L2 regularization.
- Add useful predictors or improve feature engineering.

### How to reduce overfitting

- Reduce `max_depth`.
- Increase `min_child_weight`.
- Increase `reg_lambda` or `reg_alpha` as appropriate.
- Use `subsample` and `colsample_bytree` to introduce useful randomness.
- Lower `learning_rate` and use early stopping with enough boosting rounds.
- Investigate outliers, leakage and feature quality.

Important: A smaller learning rate alone does not guarantee less overfitting if you compensate by adding unlimited boosting rounds. Tune the learning rate and number of rounds together.

# 11. XGBoost classification

XGBoost classification optimizes a classification objective, such as binary logistic loss, while adding trees sequentially. It supports regularization and gradient/Hessian-based split optimization.

## How to identify underfitting and overfitting

|                        | Underfitting     | Overfitting                                       |
| ---------------------- | ---------------- | ------------------------------------------------- |
| Training log loss      | High             | Very low                                          |
| Validation log loss    | High             | Much higher                                       |
| ROC-AUC / PR-AUC       | Weak on both     | Often stronger on train                           |
| Boosting-round pattern | Both remain poor | Validation performance deteriorates after a point |

For churn prediction, log loss and PR-AUC are particularly informative. Log loss measures probability quality, while PR-AUC focuses on the precision–recall trade-off for the positive class.

### Bias and variance

- Underfitting: high bias from weak trees, excessive regularization or insufficient boosting.
- Overfitting: high variance from overly complex trees, excessive boosting or learning noise and rare combinations of predictors.

### Dataset indicators

Underfitting

- Weak churn predictors.
- Missing network-quality, usage, billing or tenure interactions.
- Features that do not capture the relevant behavior within the prediction window.

Overfitting

- Label noise and leakage.
- Too many irrelevant features or rare categories.
- Small positive-class samples within particular subgroups.
- Large differences between training and future customer populations.

### Model and hyperparameter indicators

The main controls are the same as XGBoost regression:

- `max_depth`, `min_child_weight`
- `learning_rate`, `n_estimators`
- `reg_lambda`, `reg_alpha`
- `subsample`, `colsample_bytree`
- `gamma` (minimum loss reduction required for a split)

Too much split restriction or regularization can underfit. Too little restriction combined with aggressive boosting can overfit.

### How to reduce underfitting

- Increase tree depth modestly or reduce `min_child_weight`.
- Allow more boosting rounds if appropriate.
- Reduce excessive regularization.
- Improve the predictive feature set.
- Investigate class imbalance and the choice of evaluation metric.

### How to reduce overfitting

- Reduce depth and increase `min_child_weight`.
- Increase regularization or `gamma`.
- Use early stopping based on a suitable validation metric.
- Tune subsampling and column sampling.
- Correct leakage and remove unstable predictors.
- Use time-based or group-based validation if that reflects deployment.

For imbalanced churn data, `scale_pos_weight` can adjust the relative importance of positive examples. However, it is not an automatic overfitting fix, and it can affect probability calibration. Tune the classification threshold separately using validation data and the business cost of false positives and false negatives.

# 12. The key differences across all nine models

This is a compact revision table for interviews.

| Model                        | Typical underfitting signal           | Typical overfitting signal                          | First controls to investigate                             |
| ---------------------------- | ------------------------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Linear regression            | Systematic residual patterns          | Unstable coefficients; train–validation gap         | Feature representation, Ridge/Lasso                       |
| Logistic regression          | Poor train and validation log loss    | Train log loss much lower than validation           | `C`, penalty, features                                    |
| KNN                          | $$k$$ too large                     | $$k$$ too small                                   | `n_neighbors`, scaling, distance metric                   |
| Decision tree regression     | Shallow tree, high errors             | Deep tree, tiny leaves                              | `max_depth`, `min_samples_leaf`, pruning                  |
| Decision tree classification | Weak class separation                 | Pure tiny leaves, validation degradation            | Depth, leaf size, pruning                                 |
| Random forest regression     | Trees too restricted or features weak | Low training error, high validation error           | Leaf size, depth, `max_features`                          |
| Random forest classification | Weak train and validation metrics     | Train–validation gap, poor validation probabilities | Leaf size, depth, feature sampling                        |
| XGBoost regression           | Both errors high                      | Validation loss rises during boosting               | Depth, `min_child_weight`, regularization, early stopping |
| XGBoost classification       | Poor train and validation log loss    | Train loss falls while validation loss rises        | Depth, `min_child_weight`, regularization, early stopping |

Remember that the table describes common patterns, not universal diagnoses. Dataset shift, leakage, class imbalance and poor metric selection can mimic or obscure these patterns.

# 13. How to diagnose the root cause in practice

Suppose you train a churn model and observe the following:

| Observation                                                                | What it suggests                            | What to investigate first                                       |
| -------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| Training and validation errors are both high                               | Underfitting or poor data                   | Feature quality, model capacity, target quality                 |
| Training error is low, validation error is high                            | Overfitting or distribution mismatch        | Leakage, tree complexity, validation split                      |
| Training error decreases while validation error increases                  | Overfitting during training                 | Early stopping, regularization                                  |
| Both errors are high, but the gap is small                                 | Consistent poor fit                         | Missing signals, high bias, irreducible noise                   |
| Random forest performance is stable but XGBoost degrades after many rounds | Boosting may be overfitting                 | Early stopping, depth, regularization                           |
| Random forest and XGBoost both perform poorly on future data               | Shared data or deployment issue is possible | Temporal drift, leakage, feature availability, label definition |

## A practical debugging sequence

When I see a training–validation gap, I would follow this sequence rather than immediately changing hyperparameters.

1. Verify the evaluation setup. Check leakage, duplicate entities, target timing, preprocessing and whether the validation set reflects production.
2. Check dataset quality. Inspect missingness, outliers, label errors, class balance, feature distributions and the number of observations per important subgroup.
3. Establish a baseline. Compare with a dummy model and a simpler model to establish whether added complexity improves generalization.
4. Inspect learning curves. Compare training and validation performance as the training sample size increases. If validation performance improves with more data, additional representative examples may help.
5. Tune model capacity and regularization. Use the appropriate controls for the algorithm, keeping the validation strategy fixed.
6. Select the model using validation or cross-validation. Keep the test set untouched until the final evaluation.

## 14. Final interview question: “How would you reduce overfitting in random forest?”

Here is a concise answer you can give verbally:

> “First, I compare training and validation metrics. If training error is very low but validation error is significantly higher, I investigate leakage, data quality and distribution differences before concluding that the model is overfitting. In a random forest, I would tune `max_depth`, increase `min_samples_leaf` or `min_samples_split`, and optimize `max_features` using cross-validation. I would also remove unreliable features and add representative training data if necessary. Random forest reduces variance by averaging trees, but increasing the number of trees alone usually does not fix overfitting caused by overly complex trees.”

If the interviewer asks, “What does bias–variance trade-off have to do with this?”, add:

> “A model with high bias is too restrictive and underfits both training and unseen data. A model with high variance is sensitive to the particular training sample and tends to overfit. My objective is to find the model complexity and regularization that minimize validation error, rather than simply minimizing training error.”

The most important takeaway: diagnose the error pattern first, identify the cause second, and only then decide whether to change the dataset, feature engineering, model capacity or training procedure.