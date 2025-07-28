**Comprehensive mathematical explanation** of the **K-Means Clustering algorithm**:
---

## 📌 Overview

**K-Means** is an unsupervised clustering algorithm that aims to partition a dataset into **K non-overlapping clusters** such that each data point belongs to the cluster with the **nearest mean** (cluster center or centroid).

---

## 🧠 Objective of K-Means

The **goal** of K-Means is to **minimize the total intra-cluster variance**, or more formally, the **sum of squared distances (SSE)** between data points and their respective cluster centroids.

### ✅ Objective Function

Let:

* $X = \{x_1, x_2, \ldots, x_n\} \subset \mathbb{R}^d$ be the dataset with $n$ points in $d$-dimensional space.
* $K$ = number of clusters
* $C = \{C_1, C_2, \ldots, C_K\}$ be the clusters
* $\mu_j \in \mathbb{R}^d$ be the centroid (mean) of cluster $C_j$

The cost function (also called **distortion function** or **within-cluster sum of squares**) is:

$$
J = \sum_{j=1}^K \sum_{x_i \in C_j} \| x_i - \mu_j \|^2
$$

**We aim to minimize $J$** — the sum of squared distances from each point to the centroid of its assigned cluster.

---

## 🧮 Algorithm Steps — With Math

### 1. **Initialization**

Randomly initialize $K$ centroids $\mu_1, \mu_2, \ldots, \mu_K$ from the data.

---

### 2. **Assignment Step (E-step)**

Assign each data point $x_i$ to the **nearest centroid** based on Euclidean distance.

$$
\text{Assign } x_i \text{ to cluster } C_j \text{ such that: } j = \arg\min_k \|x_i - \mu_k\|^2
$$

This step minimizes the **distance to centroids**, clustering points based on proximity.

---

### 3. **Update Step (M-step)**

After assigning all points, update the centroids by taking the **mean of all points** in each cluster:

$$
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$

This computes the **centroid (center of mass)** of each cluster.

---

### 4. **Repeat**

Repeat **Steps 2 and 3** until:

* The centroids stop changing (converge), OR
* A maximum number of iterations is reached, OR
* Change in the objective function $J$ is less than a threshold (tolerance).

---

## 📉 Why Squared Euclidean Distance?

* Squared Euclidean distance is **differentiable**, **convex**, and computationally efficient.
* Minimizing it leads to **centroid = mean**, which is a natural and stable central location for clustering.
* Other distances (e.g., Manhattan, Cosine) can be used in variants of K-Means.

---

## 🧠 Why Does K-Means Work?

* It uses **Coordinate Descent** to minimize the objective function:

  * The **assignment step** minimizes $J$ with respect to cluster assignments.
  * The **update step** minimizes $J$ with respect to centroids.

While each step reduces or maintains the value of $J$, the overall optimization is **not convex**, and hence K-Means may converge to **local minima**.

---

## ⚠️ Limitations (Mathematically)

* Sensitive to **initialization** — different starts can lead to different results.
* Assumes **spherical clusters** with equal variance.
* Doesn’t handle **overlapping or non-convex** clusters well.
* Number of clusters $K$ must be specified manually.

---

## 📊 How to Choose $K$? (Optimal Clusters)

### 1. **Elbow Method**

Plot SSE vs. $K$, and choose the "elbow" point where the marginal gain decreases:

$$
\text{SSE}(K) = \sum_{j=1}^K \sum_{x_i \in C_j} \| x_i - \mu_j \|^2
$$

### 2. **Silhouette Score**

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

* $a(i)$ = average intra-cluster distance
* $b(i)$ = average nearest-cluster distance

High silhouette score (near 1) indicates good clustering.

---

## ✅ Summary Table

| Step           | Purpose                 | Formula                                                   |      |                                |
| -------------- | ----------------------- | --------------------------------------------------------- | ---- | ------------------------------ |
| Initialization | Random centroids        | $\mu_j \sim X$                                            |      |                                |
| Assignment     | Find closest centroid   | $j = \arg\min_k \|x_i - \mu_k\|^2$                        |      |                                |
| Update         | Mean of assigned points | ( \mu\_j = \frac{1}{                                      | C\_j | } \sum\_{x\_i \in C\_j} x\_i ) |
| Objective      | Minimize variance       | $J = \sum_{j=1}^K \sum_{x_i \in C_j} \| x_i - \mu_j \|^2$ |      |                                |


