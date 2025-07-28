# 🧠 What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a **non-parametric, instance-based** learning algorithm used for:

* **Classification**: Predict a class label
* **Regression**: Predict a continuous value

K-Nearest Neighbors is also called as a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification it performs an action on the dataset.
---

## 🧭 Core Idea

Given a new data point:

1. Compute distances from the new point to **all points** in the training data.
2. Find the **K nearest neighbors** based on a distance metric (commonly **Euclidean distance**).
3. Classification: Output the **majority class** of neighbors.
   Regression: Output the **average (or weighted average)** of their values.

---

## 📐 1. Distance Metric (Usually Euclidean)

To measure similarity, we usually compute:

### **Euclidean Distance** (for points $x$ and $x_i$):

$$
d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}
$$

Where:

* $x \in \mathbb{R}^n$ is the query/input point
* $x_i \in \mathbb{R}^n$ is a training data point

Other distance metrics (optional):

* Manhattan Distance
* Minkowski Distance
* Cosine Similarity (for text/high-dimensional)

---

# 🧾 KNN for Classification

## 🔍 Prediction Rule (for classification):

Given a query point $x$, compute distances to all training points, choose the $K$ nearest points $\{x_{(1)}, x_{(2)}, ..., x_{(K)}\}$ and their corresponding labels $\{y_{(1)}, y_{(2)}, ..., y_{(K)}\}$.

Predict the **majority class** among those labels.

### Mathematically:

$$
\hat{y} = \text{mode}\left(\{ y_{(i)} \}_{i=1}^K \right)
$$

### Optional: **Weighted KNN**

You can weight votes by **inverse distance**:

$$
w_i = \frac{1}{d(x, x_i)^2 + \epsilon}
$$

Then use weighted majority voting.

---

# 📊 KNN for Regression

## 🔍 Prediction Rule (for regression):

Instead of majority vote, take the **average of the K nearest values**:

### Simple Average:

$$
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_{(i)}
$$

### Weighted Average (optional):

$$
\hat{y} = \frac{\sum_{i=1}^{K} w_i y_{(i)}}{\sum_{i=1}^{K} w_i}, \quad \text{where } w_i = \frac{1}{d(x, x_i)^2 + \epsilon}
$$

This way, closer neighbors contribute more to the prediction.

---

## 📉 Bias-Variance Tradeoff

* **Small $K$ (e.g., 1)** → Low bias, high variance (overfitting)
* **Large $K$** → High bias, low variance (underfitting)

So, $K$ controls the **smoothness** of the prediction function.

---

# 🔍 How to Choose the Best Value of $K$

1. **Try odd values** (to avoid ties in binary classification).
2. Use **cross-validation** to find the best K.
3. Plot error vs. K to choose the one that minimizes error on validation set.

Example rule-of-thumb:

$$
K = \sqrt{n} \quad \text{where } n = \text{number of samples}
$$

But this is not optimal — use **grid search + CV** in practice.

---

## ⚖️ Differences Between Classification and Regression in KNN

| Feature              | Classification                       | Regression                          |
| -------------------- | ------------------------------------ | ----------------------------------- |
| Target Variable      | Categorical (e.g., class labels)     | Continuous (real values)            |
| Prediction Rule      | Majority vote                        | Average of neighbor values          |
| Evaluation Metric    | Accuracy, F1-score                   | MSE, RMSE, MAE                      |
| Sensitivity to Noise | Higher if class boundaries are noisy | Higher if nearby values vary widely |

---

## 🧠 Summary of Mathematical Steps

### 1. For a test point $x$:

* Compute distances $d(x, x_i)$ to all training points

### 2. Select K nearest neighbors:

* Sort distances and pick top K indices

### 3. Classification:

$$
\hat{y} = \text{mode}(\{ y_{(i)} \}_{i=1}^K)
$$

### 4. Regression:

$$
\hat{y} = \frac{1}{K} \sum_{i=1}^K y_{(i)} \quad \text{(or weighted avg)}
$$
