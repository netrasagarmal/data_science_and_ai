# 1. Why Regularization Exists

Suppose you're doing Linear Regression.

The objective is to minimize prediction error:

$$
\text{Loss} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

This only focuses on fitting training data.

Problem:

* Model may learn noise.
* Coefficients can become very large.
* Overfitting occurs.

Regularization modifies the objective:

> "Fit the data well, but don't let the model become unnecessarily complex."

---

# 2. General Form of Regularization

The loss becomes:

$$
\text{Loss} = \text{Error} + \text{Regularization Penalty}
$$

Where:

* Error term = how wrong predictions are
* Penalty term = punishment for large weights

---

# 3. Ridge Regression (L2)

### Mathematical Equation

For Linear Regression:

$$
\text{Loss} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2 + \lambda\sum_{j=1}^{p}w_j^2
$$

where:

* $w_j$ = model coefficient
* $p$ = number of features
* $\lambda$ = regularization strength

---

## Interview Explanation

The first term tries to reduce prediction error.

The second term penalizes large coefficients.

As coefficients increase:

* $w_j^2$ increases rapidly
* Penalty increases
* Optimizer prefers smaller weights

---

## Interview One-Liner

> Ridge adds a squared-weight penalty to the loss function, shrinking coefficients toward zero and reducing model variance.

---

# 4. Lasso Regression (L1)

### Mathematical Equation

$$
\text{Loss} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2 + \lambda\sum_{j=1}^{p}|w_j|
$$

Notice:

Ridge uses:

$$
w_j^2
$$

Lasso uses:

$$
|w_j|
$$

---

## Key Consequence

Because of the absolute value penalty, many coefficients become exactly:

$$
0
$$

---

## Interview One-Liner

> Lasso adds an absolute-weight penalty that can drive coefficients exactly to zero, effectively performing feature selection.

---

# 5. What Does Lambda ($\lambda$) Do?

Lambda controls penalty strength.

### $\lambda = 0$

No regularization.

Model becomes:

$$
\text{Loss} = \text{Error}
$$

Possible:

* Overfitting
* Large coefficients

### Small $\lambda$

* Slight coefficient shrinkage
* Mild regularization

### Large $\lambda$

* Strong penalty
* Very small coefficients
* Underfitting risk

---

## Interview Answer

> Lambda controls the trade-off between fitting the training data and keeping the model simple.

---

# 6. Bias-Variance Perspective

Without regularization:

* Low bias
* High variance

With regularization:

* Bias increases slightly
* Variance decreases significantly

Result:

* Better generalization

---

## Interview Answer

> Regularization intentionally introduces a small amount of bias to reduce variance and improve performance on unseen data.

---

# 7. Why Does Ridge Handle Correlated Features Better?

Suppose:

* House Area
* Carpet Area

Correlation:

$$
\rho = 0.95
$$

Without regularization:

* Coefficients can fluctuate wildly.

Ridge:

* Spreads weight across both features.
* Produces stable solutions.

---

## Interview Answer

> Ridge stabilizes coefficients in the presence of multicollinearity by shrinking correlated feature weights together.

---

# 8. Why Does Lasso Perform Feature Selection?

A practical answer:

> The L1 penalty creates optimization behavior that encourages sparse solutions, causing some coefficients to become exactly zero.

Keyword:

**Sparse Solution**

Mathematically:

$$
w_j = 0
$$

for many irrelevant features.

---

# 9. Elastic Net

Combines both L1 and L2 penalties.

### Equation

$$
\text{Loss} =
\text{Error}
+
\lambda_1\sum_{j=1}^{p}|w_j|
+
\lambda_2\sum_{j=1}^{p}w_j^2
$$

Benefits:

* L1 feature selection
* L2 stability
* Better for correlated features

---

## Interview Answer

> Elastic Net is useful when I want feature selection but also have highly correlated predictors.

---

# 10. How Do We Know Regularization Worked?

Compare performance before and after regularization.

Example:

| Metric | Before | After |
|----------|----------|----------|
| Train Accuracy | 99% | 95% |
| Validation Accuracy | 84% | 91% |

Interpretation:

* Training performance drops slightly.
* Validation performance improves.
* Overfitting reduced.

Regularization succeeded.

---

# 11. Why Should Features Be Scaled?

Suppose:

| Feature | Range |
|----------|----------|
| Salary | 10,000–100,000 |
| Age | 18–60 |

Without scaling:

Large-scale features dominate the coefficient magnitudes.

Regularization becomes unfair because penalties depend on coefficient size.

Recommended pipeline:

```text
StandardScaler
      ↓
 Ridge/Lasso
```

---

# 12. Interview Cheat Sheet (2-Minute Revision)

### What is Regularization?

Adds a penalty term to the loss function to reduce overfitting and improve generalization.

### Ridge (L2)

Penalty:

$$
\lambda\sum_{j=1}^{p}w_j^2
$$

* Shrinks coefficients
* Doesn't usually remove features
* Good for correlated features
* Reduces variance

### Lasso (L1)

Penalty:

$$
\lambda\sum_{j=1}^{p}|w_j|
$$

* Shrinks coefficients
* Can make coefficients exactly zero
* Performs feature selection
* Produces sparse models

### Elastic Net

Penalty:

$$
\lambda_1\sum_{j=1}^{p}|w_j|
+
\lambda_2\sum_{j=1}^{p}w_j^2
$$

* Feature selection + stability
* Useful with correlated features

### Lambda

* Small $\lambda$ → weak regularization
* Large $\lambda$ → strong regularization
* $\lambda = 0$ → no regularization

### When to Use What?

| Situation | Choice |
|------------|------------|
| Many irrelevant features | Lasso |
| Correlated features | Ridge |
| Need feature selection + correlated features | Elastic Net |
| Model unstable | Ridge |
| Need interpretability | Lasso |

### Common Interview Summary

> I would diagnose overfitting using train-vs-validation performance, apply regularization if needed, tune $\lambda$ using cross-validation, and compare Ridge, Lasso, and Elastic Net depending on feature sparsity requirements and feature correlations.


</details>