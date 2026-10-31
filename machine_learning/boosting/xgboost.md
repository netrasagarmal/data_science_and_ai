## XGBoost

Yes. The easiest way to understand **XGBoost** is:

> **XGBoost is still gradient boosting, but it makes the tree-building process much more sophisticated and adds explicit regularization, second-order optimization, efficient split finding, parallelism, and handling of sparse/missing data.**

I'll use the same salary example first, then connect each difference to the actual XGBoost mathematics.

---

## 1. First: Gradient Boosting vs XGBoost

Think of standard Gradient Boosting as:

```text
Base prediction
      ↓
Calculate residual / negative gradient
      ↓
Build decision tree on residuals
      ↓
Multiply by learning rate
      ↓
Add tree to ensemble
      ↓
Repeat
```

XGBoost keeps this basic idea:

```text
Current prediction
      ↓
Calculate gradient + Hessian
      ↓
Find the best tree structure
      ↓
Calculate optimal leaf weights
      ↓
Regularize the tree
      ↓
Add tree × learning rate
      ↓
Repeat
```

The major difference is:

### Gradient Boosting

Mostly asks:

> "What residual should my next tree learn?"

### XGBoost

Asks:

> "Given the current predictions, gradients, and curvature of the loss, what tree structure and leaf values will produce the greatest regularized reduction in objective?"

That's a much more optimization-oriented approach.

---
Let's take the same salary prediction problem and walk through XGBoost from beginning to end, just as if we were implementing it manually.

We'll track exactly:

1. What data enters the model.
2. How the base prediction is calculated.
3. How gradients and Hessians are calculated.
4. What Tree 1 receives and what it produces.
5. How predictions are updated.
6. What goes into Tree 2 and Tree 3.
7. How the final salary prediction is calculated.

The most important concept: Tree 2 does not receive the output of Tree 1 as its training target. It receives newly calculated gradients and Hessians based on the updated predictions.

# 2.1. Our salary dataset

We'll use the exact same 10 records.

| Row | Experience | Degree | Actual salary (₹L) |
| --- | ---------- | ------ | ------------------ |
| 1   | 1          | UG     | 45                 |
| 2   | 2          | UG     | 50                 |
| 3   | 3          | PG     | 58                 |
| 4   | 4          | PG     | 65                 |
| 5   | 5          | PG     | 70                 |
| 6   | 6          | PhD    | 82                 |
| 7   | 7          | PhD    | 88                 |
| 8   | 8          | PhD    | 95                 |
| 9   | 9          | PG     | 78                 |
| 10  | 10         | PhD    | 110                |

We'll use these XGBoost settings throughout:

| Parameter                         | Value         | Meaning                             |
| --------------------------------- | ------------- | ----------------------------------- |
| Objective                         | Squared error | Predict salary                      |
| Learning rate ($\eta$)        | 0.1           | Apply 10% of each tree's correction |
| L2 regularization ($\lambda$) | 10            | Control leaf weights                |
| Split penalty ($\gamma$)      | 1             | Penalize unnecessary splits         |
| Number of trees                   | 3             | Three boosting rounds               |

For clarity, I'll use simple one-split trees and specify their splits manually to demonstrate the calculations. In actual training, XGBoost evaluates candidate splits and chooses those with the best gain.

# 2.2. Step 1 — Calculate the base prediction

Before building any tree, XGBoost needs an initial prediction.

For squared-error regression, we'll use the mean salary:

$$ F_0=\frac{45+50+58+65+70+82+88+95+78+110}{10} $$

$$ \boxed{F_0=74.1} $$

This means that before building any trees, the model predicts ₹74.1L for every employee.

Initial prediction for all 10 employees

# ₹74.1L

Same prediction for every row, before any tree is built.

# 2.3. Step 2 — Calculate gradients and Hessians

Now compare the actual salary with the initial prediction.

For this example, our loss function is:

$$ L=\frac12(y-\hat y)^2 $$

The derivatives are:

$$ g_i=\hat y_i-y_i $$

$$ h_i=1 $$

Let's calculate them for all 10 records.

| Row | Actual y | Prediction | Gradient g | Hessian h |
| --- | -------- | ---------- | ---------- | --------- |
| 1   | 45       | 74.1       | +29.1      | 1         |
| 2   | 50       | 74.1       | +24.1      | 1         |
| 3   | 58       | 74.1       | +16.1      | 1         |
| 4   | 65       | 74.1       | +9.1       | 1         |
| 5   | 70       | 74.1       | +4.1       | 1         |
| 6   | 82       | 74.1       | -7.9       | 1         |
| 7   | 88       | 74.1       | -13.9      | 1         |
| 8   | 95       | 74.1       | -20.9      | 1         |
| 9   | 78       | 74.1       | -3.9       | 1         |
| 10  | 110      | 74.1       | -35.9      | 1         |

Understand the signs:

- Positive gradient: prediction is too high; we need to decrease it.
- Negative gradient: prediction is too low; we need to increase it.
- Hessian: for this particular squared-error loss, every row has a value of 1.

Important: What enters Tree 1?

The training algorithm has:

- Original features: experience and degree.
- Gradient values: $g_i$.
- Hessian values: $h_i$.

The tree uses the original features to create splits, while gradients and Hessians tell it how valuable those splits and their leaf corrections are.

# 2.4. Step 3 — Build Tree 1

Let's say the first tree creates this split:

### Tree 1 — Root split

Experience ≤ 5.5?


### Left leaf

Experience ≤ 5.5

Rows 1–5

# -5.50

Calculated leaf weight

### Right leaf

Experience > 5.5

Rows 6–10
# +5.50

Calculated leaf weight

Now, how did XGBoost calculate these two leaf values?

#### A. Calculate gradient and Hessian sums for the left leaf.

Rows 1–5:

$$ G_L=29.1+24.1+16.1+9.1+4.1=82.5 $$

$$ H_L=1+1+1+1+1=5 $$

#### B. Calculate gradient and Hessian sums for the right leaf.

Rows 6–10:

$$ G_R=-7.9-13.9-20.9-3.9-35.9=-82.5 $$

$$ H_R=5 $$

#### C. Calculate the optimal leaf weights.

XGBoost uses:

$$ w_j=-\frac{G_j}{H_j+\lambda} $$

Left leaf:

$$ w_L=-\frac{82.5}{5+10}=-5.5 $$

Right leaf:

$$ w_R=-\frac{-82.5}{5+10}=+5.5 $$

**Tree 1 output**

| Rows | Tree 1 output |
| ---- | ------------- |
| 1–5  | -5.50         |
| 6–10 | +5.50         |

Notice that these are not salary predictions. They are corrections that the tree contributes to the current predictions.

#### D. Update the predictions using the learning rate.

$$ \hat y^{(1)}=74.1+0.1\times T_1(X) $$

For rows 1–5:

$$ 74.1+(0.1\times-5.5)=73.55 $$

For rows 6–10:

$$ 74.1+(0.1\times5.5)=74.65 $$

**Predictions after Tree 1**
|Row|Prediction|
|-|-|
| Rows 1–5 | 73.55|
|Rows 6–10|74.65|

# 2.5. Step 4 — Prepare the input for Tree 2

This is where you need to pay close attention.

We do not reuse the original gradients.

We calculate new gradients using the updated predictions from Tree 1.

The actual salary remains unchanged, but the prediction has changed.

$$ g_i^{(2)}=\hat y_i^{(1)}-y_i $$

$$ h_i^{(2)}=1 $$

Let's calculate the new values.

| Row | Actual salary | New prediction | New gradient | Hessian |
| --- | ------------- | -------------- | ------------ | ------- |
| 1   | 45            | 73.55          | +28.55       | 1       |
| 2   | 50            | 73.55          | +23.55       | 1       |
| 3   | 58            | 73.55          | +15.55       | 1       |
| 4   | 65            | 73.55          | +8.55        | 1       |
| 5   | 70            | 73.55          | +3.55        | 1       |
| 6   | 82            | 74.65          | -7.35        | 1       |
| 7   | 88            | 74.65          | -13.35       | 1       |
| 8   | 95            | 74.65          | -20.35       | 1       |
| 9   | 78            | 74.65          | -3.35        | 1       |
| 10  | 110           | 74.65          | -35.35       | 1       |

**What exactly goes into Tree 2?**

- Original features: experience and degree.
- Updated gradients from Tree 1.
- Updated Hessians (still 1 in this example).

**Tree 2 now learns how to correct the remaining errors.**

# 2.6. Step 5 — Build Tree 2

For our illustration, let's use this split:

### Tree 2 — Root split

Experience ≤ 2.5?

### Left leaf

Rows 1–2

# -4.3417

### Right leaf

Rows 3–10

# +2.8944

Let's calculate the leaf weights.

#### A. Left leaf — rows 1–2

$$ G_L=28.55+23.55=52.1 $$

$$ H_L=2 $$

$$ w_L=-\frac{52.1}{2+10} =\boxed{-4.3417} $$

#### B. Right leaf — rows 3–10

$$ G_R=-52.1,\qquad H_R=8 $$

$$ w_R=-\frac{-52.1}{8+10} =\boxed{+2.8944} $$

#### C. Update predictions again.

$$ \hat y^{(2)}=\hat y^{(1)}+0.1\times T_2(X) $$

| Rows | Previous prediction | Tree 2 correction | New prediction |
| ---- | ------------------- | ----------------- | -------------- |
| 1–2  | 73.55               | -0.4342           | 73.1158        |
| 3–5  | 73.55               | +0.2894           | 73.8394        |
| 6–10 | 74.65               | +0.2894           | 74.9394        |

Now our model has a different prediction for three groups of employees.

# 2.7. Step 6 — Prepare the input for Tree 3

Again, calculate new gradients using the predictions after Tree 2.

$$ g_i^{(3)}=\hat y_i^{(2)}-y_i $$

$$ h_i^{(3)}=1 $$

| Row | Actual salary | Prediction after Tree 2 | New gradient | Hessian |
| --- | ------------- | ----------------------- | ------------ | ------- |
| 1   | 45            | 73.1158                 | +28.1158     | 1       |
| 2   | 50            | 73.1158                 | +23.1158     | 1       |
| 3   | 58            | 73.8394                 | +15.8394     | 1       |
| 4   | 65            | 73.8394                 | +8.8394      | 1       |
| 5   | 70            | 73.8394                 | +3.8394      | 1       |
| 6   | 82            | 74.9394                 | -7.0606      | 1       |
| 7   | 88            | 74.9394                 | -13.0606     | 1       |
| 8   | 95            | 74.9394                 | -20.0606     | 1       |
| 9   | 78            | 74.9394                 | -3.0606      | 1       |
| 10  | 110           | 74.9394                 | -35.0606     | 1       |

The process is exactly the same, but the gradient values have changed again.

# 2.8. Step 7 — Build Tree 3

For our illustration, let's use the split:

**Tree 3 — Root split**

Experience ≤ 7.5?

### Left leaf

Rows 1–7

# -3.5076

### Right leaf

Rows 8–10

# +4.4755

Calculate the leaf values.

#### A. Left leaf — rows 1–7

$$ G_L=28.1158+23.1158+\cdots-13.0606 $$

$$ G_L=59.6289,\quad H_L=7 $$

$$ w_L=-\frac{59.6289}{7+10} =\boxed{-3.5076} $$

#### B. Right leaf — rows 8–10

$$ G_R=-20.0606-3.0606-35.0606 $$

$$ G_R=-58.1817,\quad H_R=3 $$

$$ w_R=-\frac{-58.1817}{3+10} =\boxed{+4.4755} $$

#### C. Update predictions.

$$ \hat y^{(3)}=\hat y^{(2)}+0.1\times T_3(X) $$

| Rows | Previous prediction | Tree 3 correction | Final prediction |
| ---- | ------------------- | ----------------- | ---------------- |
| 1–2  | 73.1158             | -0.3508           | 72.7651          |
| 3–5  | 73.8394             | -0.3508           | 73.4887          |
| 6–7  | 74.9394             | -0.3508           | 74.5887          |
| 8–10 | 74.9394             | +0.4476           | 75.3870          |

# 2.9. Step 8 — Final prediction after all 3 trees

Now combine the outputs of all three trees.

**The final formula is:**

$$ \boxed{ \hat y=F_0+\eta T_1(X)+\eta T_2(X)+\eta T_3(X) } $$

Let's calculate it for two employees.

**Employee 1 — Experience 1, UG**

- Base prediction = 74.1
- Tree 1 output = -5.5
- Tree 2 output = -4.3417
- Tree 3 output = -3.5076

$$ \hat y=74.1+0.1(-5.5)+0.1(-4.3417)+0.1(-3.5076) $$

$$ \boxed{\hat y=72.7651\text{ L}} $$

Actual salary = ₹45L.

**Employee 8 — Experience 8, PhD**

- Base prediction = 74.1
- Tree 1 output = +5.5
- Tree 2 output = +2.8944
- Tree 3 output = +4.4755

$$ \hat y=74.1+0.1(5.5)+0.1(2.8944)+0.1(4.4755) $$

$$ \boxed{\hat y=75.3870\text{ L}} $$

Actual salary = ₹95L.

Here are the final predictions for all 10 records:

| Row | Actual salary | Final predicted salary |
| --- | ------------- | ---------------------- |
| 1   | 45            | 72.77                  |
| 2   | 50            | 72.77                  |
| 3   | 58            | 73.49                  |
| 4   | 65            | 73.49                  |
| 5   | 70            | 73.49                  |
| 6   | 82            | 74.59                  |
| 7   | 88            | 74.59                  |
| 8   | 95            | 75.39                  |
| 9   | 78            | 75.39                  |
| 10  | 110           | 75.39                  |

Notice that three trees are not necessarily enough to produce good predictions. This is a deliberately small demonstration of the mechanics, not a trained production-quality model.

# 2.10. The entire XGBoost process in one flow

This is the flow I would memorize for an interview.

1. Original dataset

   Features (Experience, Degree) + Actual salary

2. Base prediction

   Mean salary = 74.1

3. Calculate gradients and Hessians

   $g=\hat y-y,\quad h=1$

4. Tree 1

   Input: Original features + initial gradients and Hessians

   Output: Tree structure and leaf weights

   Update predictions using learning rate

5. Recalculate gradients and Hessians

   Using predictions after Tree 1

6. Tree 2

   Input: Original features + updated gradients and Hessians

   Output: New tree and leaf weights

   Update predictions again

7. Recalculate gradients and Hessians

   Using predictions after Tree 2

8. Tree 3

   Input: Original features + latest gradients and Hessians

   Output: New tree and leaf weights

   Update predictions again

9. Final prediction

10. Base score + learning-rate-scaled outputs of all 3 trees

## One last important distinction: where does split gain come in?

You might have noticed that we calculated leaf weights, but not yet explained how XGBoost decides whether to make a split.

It evaluates candidate splits using the gain formula:

$$ Gain=\frac12\left[ \frac{G_L^2}{H_L+\lambda} +\frac{G_R^2}{H_R+\lambda} -\frac{G^2}{H+\lambda} \right]-\gamma $$

This happens while constructing each tree, before finalizing its structure and leaf values.

**So, for every tree, the process is:**

1. Take the current gradients and Hessians.
2. Evaluate possible feature splits.
3. Calculate the gain of each split.
4. Choose worthwhile splits, considering the regularization penalty.
5. Calculate the optimal leaf weights.
6. Produce the tree output.
7. Update predictions.

> Remember this distinction: Gradients and Hessians are recalculated for every boosting round. The original feature dataset remains the same. The model's predictions change after each tree, and those updated predictions drive the next round.
---

# 12. How does XGBoost decide whether a split is good?

This is another major difference.

Suppose we have:

```text
Parent node
       |
       | split
       ↓
  Left      Right
```

XGBoost calculates:

$$
G_L,H_L
$$

and:

$$
G_R,H_R
$$

The approximate split gain is:

$$ 
\boxed{ Gain = \frac12
\left[
\frac{G_L^2}{H_L+\lambda} +
\frac{G_R^2}{H_R+\lambda} -
\frac{G^2}{H+\lambda}
\right] - \gamma
}
$$

where:

$$
G=G_L+G_R
$$

and:

$$
H=H_L+H_R
$$

This formula is **central to understanding XGBoost**.

---

# 13. What is XGBoost actually doing while building the tree?

Suppose it considers:

```text
Experience <= 1.5
Experience <= 2.5
Experience <= 3.5
Experience <= 4.5
...
```

For each candidate split it calculates:

```text
left:
    sum gradients
    sum Hessians

right:
    sum gradients
    sum Hessians

        ↓

calculate split gain

        ↓

choose split with highest positive gain
```

So XGBoost isn't just blindly growing a CART tree.

It is using the **gradient statistics of the current boosting iteration to select splits**.

---

# 14. Gradient Boosting vs XGBoost tree construction

This is a useful interview comparison.

### Traditional Gradient Boosting

Conceptually:

```text
Calculate residuals
      ↓
Give residuals to decision-tree algorithm
      ↓
Tree finds splits that reduce residual squared error
      ↓
Tree predicts residual
```

### XGBoost

```text
Calculate gradients + Hessians
          ↓
For every candidate split
          ↓
Calculate G and H
          ↓
Calculate regularized gain
          ↓
Choose best split
          ↓
Calculate optimal regularized leaf weights
```

So XGBoost incorporates the optimization objective directly into tree construction.

---

# 15. Parallel processing

Another major XGBoost advantage is computational efficiency.

A naive tree-building process might look like:

```text
Feature 1
   ↓
try splits

Feature 2
   ↓
try splits

Feature 3
   ↓
try splits
```

XGBoost can perform much of the split evaluation in parallel.

For example:

```text
              Training data
                    |
       ┌────────────┼────────────┐
       ↓            ↓            ↓
   Feature 1    Feature 2    Feature 3
       ↓            ↓            ↓
   histogram     histogram     histogram
       ↓            ↓            ↓
     gains        gains        gains
       └────────────┼────────────┘
                    ↓
               best split
```

Modern XGBoost implementations use highly optimized tree-building algorithms, including histogram-based methods, to reduce the amount of computation and memory required.

---

# 16. Why can split finding be parallelized?

Suppose we have:

```text
1 million rows
500 features
```

For each feature, we need information such as:

$$
\sum g_i
$$

and:

$$
\sum h_i
$$

for candidate split regions.

These calculations can be performed efficiently using parallel computation.

So rather than:

```text
feature 1 → finish
feature 2 → finish
feature 3 → finish
...
```

we can process feature/split statistics in parallel.

This doesn't mean:

> "All trees are trained simultaneously."

That's an important distinction.

### Trees within the boosting sequence are dependent.

Tree 2 needs predictions from Tree 1.

Therefore:

```text
Tree 1
  ↓
Tree 2
  ↓
Tree 3
```

cannot simply all be trained independently.

The parallelism is primarily **within the construction of each tree**, especially feature/split statistics and histogram operations.

---

# 17. Sparsity-aware splitting

This is particularly important in real-world enterprise data.

Imagine:

```text
Customer data

age
income
5G_usage
international_roaming
...
```

You could have:

```text
5G_usage = 0
international_roaming = missing
special_offer = missing
```

Large ML datasets can have many:

- missing values
- zero values
- one-hot encoded features
- sparse matrices

A naive algorithm may waste huge amounts of computation processing all those zeros/missing values.

XGBoost is designed to exploit sparsity.

---

# 18. What does sparsity-aware mean?

Suppose:

```text
Feature = international_roaming
```

Dataset:

| Customer | Feature |
|---:|---:|
| 1 | 1 |
| 2 | 0 |
| 3 | 0 |
| 4 | missing |
| 5 | 0 |
| 6 | 1 |
| 7 | missing |
| 8 | 0 |

Most values are zero/missing.

Instead of treating every zero as an ordinary numerical value and doing unnecessary work, XGBoost can efficiently handle the sparse representation.

For missing values, XGBoost learns a **default direction** for the missing values when evaluating a split.

Conceptually:

```text
Feature <= threshold?

             Split
               |
       ┌───────┴────────┐
       ↓                ↓
     left             right

Missing?
   ↓
learn whether
left or right
gives better gain
```

So the algorithm can learn:

```text
missing → left
```

or:

```text
missing → right
```

depending on which gives better objective improvement.

---

# 19. Why is this valuable?

Imagine a telecom dataset with:

```text
500 million customers
5,000 features
```

where many features represent things such as:

```text
roaming_usage
international_calls
OTT_subscription
device_type
special_service
```

Many customers may have:

```text
0 usage
```

or missing values.

Efficient sparse handling can significantly reduce the computational burden.

This is one reason XGBoost became extremely popular for large tabular datasets.

---

# 20. Put everything together

Now compare the two algorithms using our salary example.

### Gradient Boosting

```text
Base prediction
74.1
     ↓
Residuals
y - prediction
     ↓
Train regression tree
     ↓
Tree predicts residual
     ↓
× learning rate
     ↓
Add correction
     ↓
Repeat
```

### XGBoost

```text
Base prediction
74.1
     ↓
Calculate:
    gradient g
    Hessian h
     ↓
For candidate splits:
    calculate G and H
     ↓
Calculate regularized gain
     ↓
Choose best split
     ↓
Calculate optimal leaf weight
     ↓
Apply regularization
     ↓
× learning rate
     ↓
Update prediction
     ↓
Repeat
```

---

# 21. The same example in one table

| Aspect | Gradient Boosting | XGBoost |
|---|---|---|
| Basic idea | Sequential error correction | Sequential error correction |
| Tree target | Residual / negative gradient | Gradient + Hessian used in optimization |
| First derivative | Used conceptually | **Explicitly used** |
| Second derivative | Generally not used | **Explicitly used** |
| Leaf value | Often mean residual | **Optimal regularized leaf weight** |
| Regularization | More limited/basic depending on implementation | **Explicit L1/L2 + tree complexity regularization** |
| Split selection | Tree impurity/error reduction | **Gradient/Hessian-based regularized gain** |
| Learning rate | Yes | Yes |
| Number of trees | Yes | Yes |
| Parallelization | More limited in classic implementations | **Highly optimized/parallel tree construction** |
| Sparse data | Basic handling depends on implementation | **Sparsity-aware algorithms** |
| Missing values | Depends on implementation | **Learns default split direction** |
| Computation | Can be slower | Highly optimized |
| Main philosophy | Fit residuals | Optimize regularized objective |

---

# 22. One subtle point: XGBoost is still gradient boosting

Don't explain them as completely different algorithms.

That's incorrect.

The relationship is:

```text
                Boosting
                   │
          Gradient Boosting
                   │
              XGBoost
```

More accurately, XGBoost is an **optimized, regularized gradient boosting framework** that uses second-order information and efficient systems techniques.

So if an interviewer asks:

> "What's the difference between Gradient Boosting and XGBoost?"

Don't say:

> "Gradient boosting uses residuals while XGBoost uses gradients."

That's misleading because gradient boosting itself is based on gradients.

Instead say:

> **"XGBoost follows the same additive gradient boosting framework, but differs in how aggressively it optimizes the objective. It uses both first- and second-order derivatives, explicitly regularizes tree complexity and leaf weights, uses gradient/Hessian-based split gain, and introduces highly optimized parallel and sparsity-aware tree construction."**

---

# 23. And this connects directly to your previous confusion

You asked earlier:

> "What exactly is the tree trying to achieve?"

For **Gradient Boosting**:

$$
\boxed{\text{Tree learns the current negative gradient/residual}}
$$

For **XGBoost**:

$$
\boxed{
\text{Tree structure + leaf values are chosen to maximize regularized objective improvement}
}
$$

That's the key conceptual upgrade.

The progression is:

```text
Gradient Boosting:

"What errors should I correct?"
             ↓
      residuals/gradients
             ↓
           tree


XGBoost:

"What correction and tree structure
will reduce my loss the most,
while controlling model complexity?"
             ↓
       gradients + Hessians
             ↓
       split gain + regularization
             ↓
       optimal leaf weights
             ↓
            tree
```

### If you're preparing for an experienced AI/ML interview, remember these four XGBoost keywords:

**1. First derivative → direction**

**2. Second derivative → curvature**

**3. Regularized gain → whether a split is worth taking**

**4. Sparsity-aware → efficiently handle sparse/missing features**

And one final distinction:

> **Parallelization speeds up construction of an individual tree; it does not remove the sequential dependency between boosting rounds.**

---

# XGBoost First and Second Order Derivative Mathematics:

Let's derive both derivatives from the beginning, step by step, using our same salary prediction example.

The important thing is to understand three things:

1. Why does the first derivative become $\hat y-y$?
2. Why does the second derivative become $1$?
3. What do these two values actually tell XGBoost when building a decision tree?

# 1. Start with the loss function

In our salary example, suppose:

- Actual salary: $y=45$
- Current predicted salary: $\hat y=74.1$

We are using the squared error loss function:

$$ \boxed{L(y,\hat y)=\frac{1}{2}(y-\hat y)^2} $$

Why do we use this?

Because we want to penalize the difference between actual salary and predicted salary.

For our example:

$$ L(45,74.1)=\frac12(45-74.1)^2 $$

$$ =\frac12(-29.1)^2 $$

$$ =\frac12(846.81)=423.405 $$

This is the loss for this one observation.

Our objective is to reduce this loss by improving the prediction.

> Important: We use the factor $1/2$ simply to make differentiation easier. It cancels the 2 that appears when differentiating a square. Without it, the derivatives would be $2(\hat y-y)$ and $2$, respectively.

# 2. First derivative — where does $\hat y-y$ come from?

We start with:

$$ L(y,\hat y)=\frac12(y-\hat y)^2 $$

We want to calculate:

$$ g_i=\frac{\partial L}{\partial \hat y} $$

The symbol $\partial$ means partial derivative.

In this calculation:

- $y$ is treated as a constant.
- $\hat y$ is the variable we are differentiating with respect to.

**Let's break it down.**

**Step 1: Identify the outer and inner functions.**

We have:

$$ L=\frac12(y-\hat y)^2 $$

This is a function inside another function.

Let:

$$ u=y-\hat y $$

Therefore:

$$ L=\frac12u^2 $$

**Step 2: Differentiate the outer function.**

Using the power rule:

$$ \frac{d}{du}\left(\frac12u^2\right) $$

$$ =\frac12\times 2u $$

$$ =u $$

**Step 3: Differentiate the inner function.**

Remember:

$$ u=y-\hat y $$

Differentiate with respect to $\hat y$:

$$ \frac{du}{d\hat y} = \frac{d}{d\hat y}(y-\hat y) $$

Since $y$ is constant:

$$ =0-1 $$

$$ =-1 $$

**Step 4: Apply the chain rule.**

The chain rule says:

$$ \frac{dL}{d\hat y} = \frac{dL}{du} \times \frac{du}{d\hat y} $$

Substitute:

$$ =u\times(-1) $$

Since $u=y-\hat y$:

$$ =(y-\hat y)(-1) $$

Therefore:

$$ \boxed{g_i=\hat y-y} $$

That's exactly where the first derivative comes from!

## Step 2.1: Verify using our actual numbers

We know:

$$ \hat y=74.1,\quad y=45 $$

Therefore:

$$ g_i=\hat y-y $$

$$ g_i=74.1-45 $$

First derivative (gradient)

# +29.1

The positive gradient tells us that the current prediction is too high. To reduce the loss, we should move the prediction downward.

For another example, suppose:

$$ y=110,\quad \hat y=74.1 $$

Then:

$$ g_i=74.1-110=-35.9 $$

First derivative (gradient)

# -35.9

The negative gradient tells us that the current prediction is too low. To reduce the loss, we should move the prediction upward.

> Remember: The gradient tells us the direction in which the loss increases. We move in the opposite direction to reduce the loss.

# 3. Second derivative — where does 1 come from?

Now let's differentiate the first derivative again.

We already calculated:

$$ g_i=\hat y-y $$

The second derivative is:

$$ h_i=\frac{\partial^2 L}{\partial\hat y^2} $$

By definition, this means differentiating the first derivative with respect to the same variable, $\hat y$.

So:

$$ h_i=\frac{\partial}{\partial\hat y} \left(\frac{\partial L}{\partial\hat y}\right) $$

Substitute our first derivative:

$$ h_i=\frac{\partial}{\partial\hat y}(\hat y-y) $$

Now differentiate each term separately.

First term:

$$ \frac{\partial \hat y}{\partial\hat y}=1 $$

Why? Because the derivative of a variable with respect to itself is 1.

Second term:

$$ \frac{\partial y}{\partial\hat y}=0 $$

Why? Because actual salary $y$ is constant. It doesn't change when we change our predicted salary.

Therefore:

$$ h_i=1-0 $$

$$ \boxed{h_i=1} $$

That's it!

## Step 3.1: Verify using the original loss function

We can also calculate the second derivative directly.

Original loss:

$$ L=\frac12(y-\hat y)^2 $$

First derivative:

$$ \frac{\partial L}{\partial\hat y} =\hat y-y $$

Second derivative:

$$ \frac{\partial^2L}{\partial\hat y^2} = \frac{\partial}{\partial\hat y}(\hat y-y) $$

$$ =1-0 $$

$$ \boxed{h_i=1} $$

Both methods give the same result.

# 4. But what does the second derivative actually mean?

This is more important than just knowing how to calculate it.

Think about a loss curve.

- X-axis: predicted salary.
- Y-axis: loss.

For squared error, the loss curve is a parabola.

**Visual intuition:** squared-error loss

**Conceptual illustration:** the minimum occurs when predicted salary equals actual salary.

> The first derivative tells us the slope at a particular point.

> The second derivative tells us how quickly that slope changes.

In mathematical terms:

| Derivative        | Meaning                     |
| ----------------- | --------------------------- |
| First derivative  | Slope/direction of the loss |
| Second derivative | Curvature of the loss       |

For squared error:

$$ h_i=1 $$

This means the loss has constant curvature. No matter whether our prediction is 50, 70 or 100, the curvature remains the same.

The second derivative doesn't tell us whether the prediction is correct. It tells us how the loss behaves around the current prediction.

# 5. Put both derivatives together using our salary dataset

Let's calculate the gradient and Hessian for all 10 rows.

Our initial prediction is 74.1 for every person.

| Row | Actual salary | Prediction | Gradient $g=\hat y-y$ | Hessian $h$ |
| --- | ------------- | ---------- | ------------------------- | --------------- |
| 1   | 45            | 74.1       | +29.1                     | 1               |
| 2   | 50            | 74.1       | +24.1                     | 1               |
| 3   | 58            | 74.1       | +16.1                     | 1               |
| 4   | 65            | 74.1       | +9.1                      | 1               |
| 5   | 70            | 74.1       | +4.1                      | 1               |
| 6   | 82            | 74.1       | -7.9                      | 1               |
| 7   | 88            | 74.1       | -13.9                     | 1               |
| 8   | 95            | 74.1       | -20.9                     | 1               |
| 9   | 78            | 74.1       | -3.9                      | 1               |
| 10  | 110           | 74.1       | -35.9                     | 1               |

Notice:

- Rows 1–5: positive gradients because we overpredicted salary.
- Rows 6–10: negative gradients because we underpredicted salary.
- Every Hessian is 1 because we're using squared error.

# 6. Now why does XGBoost need both derivatives?

This is where the actual XGBoost algorithm comes into the picture.

Suppose XGBoost considers putting Rows 1 and 2 into the same decision-tree leaf.

Their values are:

| Row   | Gradient | Hessian |
| ----- | -------- | ------- |
| 1     | 29.1     | 1       |
| 2     | 24.1     | 1       |
| Total | 53.2     | 2       |

XGBoost aggregates them:

$$ G=\sum_i g_i=53.2 $$

$$ H=\sum_i h_i=2 $$

It then calculates the optimal leaf weight:

$$ \boxed{w^\*=-\frac{G}{H+\lambda}} $$

Suppose L2 regularization:

$$ \lambda=10 $$

Then:

$$ w^\*=-\frac{53.2}{2+10} $$

$$ \boxed{w^\*=-4.433} $$

This is the correction XGBoost wants to apply to that leaf before learning-rate scaling.

Why are both derivatives needed?

- $G$ tells us the direction and magnitude of the accumulated loss gradient.
- $H$ tells us the accumulated curvature, helping determine the appropriate correction size.
- $\lambda$ prevents excessively large corrections.

The learning rate then controls how much of that correction is applied.

# 7. Where do first and second derivatives come from mathematically in XGBoost?

XGBoost uses something called a second-order Taylor expansion.

This is the mathematical foundation of its optimization.

Suppose our current prediction is:

$$ \hat y $$

And we want to change it by a small amount:

$$ w $$

The new prediction becomes:

$$ \hat y\_{\text{new}}=\hat y+w $$

Instead of recalculating the complete loss for every possible change, XGBoost approximates the new loss using:

$$ \boxed{ L(\hat y+w)\approx L(\hat y)+gw+\frac12hw^2 } $$

Let's understand each term.

| Term              | Meaning                         |
| ----------------- | ------------------------------- |
| $L(\hat y)$   | Current loss                    |
| $gw$          | Effect of the first derivative  |
| $\frac12hw^2$ | Effect of the second derivative |
| $w$           | Proposed prediction correction  |

Notice that the approximation uses both the slope and curvature of the loss.

Now XGBoost can optimize this expression to find the best correction.

Differentiate with respect to $w$:

$$ \frac{\partial L}{\partial w} = g+hw $$

Set it equal to zero:

$$ g+hw=0 $$

Rearrange:

$$ hw=-g $$

Therefore:

$$ \boxed{w=-\frac{g}{h}} $$

This is the basic second-order optimization step.

When we aggregate multiple observations into a leaf and add L2 regularization, it becomes:

$$ \boxed{w^\*=-\frac{G}{H+\lambda}} $$

This is how the first and second derivatives ultimately influence the leaf values in XGBoost.

# 8. What happens if we use a different loss function?

This is where the second derivative becomes even more interesting.

For squared error, Hessian is always 1.

But consider binary classification with logistic loss.

The derivatives become:

$$ g_i=p_i-y_i $$

$$ h_i=p_i(1-p_i) $$

where $p_i$ is the predicted probability.

For example:

| Actual class | Predicted probability | Gradient | Hessian |
| ------------ | --------------------- | -------- | ------- |
| 1            | 0.90                  | -0.10    | 0.09    |
| 1            | 0.60                  | -0.40    | 0.24    |
| 0            | 0.80                  | +0.80    | 0.16    |
| 0            | 0.20                  | +0.20    | 0.16    |

Notice that Hessian is no longer constant.

This is one reason second-order information is useful: different observations have different local curvature, and XGBoost takes this into account when calculating leaf weights and split gains.

# 9. Final interview-ready understanding

If an interviewer asks you:

**Why does XGBoost use first and second derivatives?**

You can answer:

"XGBoost uses first-order and second-order derivatives of the loss function with respect to predictions. The first derivative tells us the direction and magnitude of the loss gradient, while the second derivative represents the curvature of the loss. XGBoost uses both through a second-order Taylor approximation to calculate optimal regularized leaf weights and evaluate split gains. This makes its optimization more informed than simply fitting trees to residuals."

## Quick revision notes

| Concept             | Formula                   | Meaning                     |
| ------------------- | ------------------------- | --------------------------- |
| Loss                | $\frac12(y-\hat y)^2$ | Prediction error            |
| First derivative    | $\hat y-y$            | Slope of loss               |
| Second derivative   | $1$                   | Curvature of loss           |
| Aggregate gradient  | $G=\sum g_i$          | Total gradient in a leaf    |
| Aggregate Hessian   | $H=\sum h_i$          | Total curvature in a leaf   |
| Optimal leaf weight | $-G/(H+\lambda)$      | Best regularized correction |

One final distinction to remember: The second derivative being 1 is specific to our chosen squared-error loss with the $1/2$ factor. It is not always 1 in XGBoost. The derivatives depend on the loss function being optimized.