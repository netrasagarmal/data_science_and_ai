# 🌳 **Decision Tree Regressor — Simple Notes**

### 🧩 What is it?

A **Decision Tree Regressor** is a machine learning model that **predicts continuous (numeric) values** — like house prices, temperature, or sales —
by **splitting data into smaller and smaller groups** based on feature values.

Each split tries to make the data in each group as **similar (pure)** as possible in terms of the **target value**.

---

### 🎯 **Goal**

Instead of predicting categories (like "Yes"/"No"),
it predicts **numbers** (like `250,000`, `72.5`, etc.)

So the goal is:

> Split the data such that each leaf node contains data points with **similar numeric values**.

---

### 🧠 **How It Works — Step by Step**

1. **Start with all data** at the root node.
2. For every possible feature and split point:

   * Check **how well the split reduces variation** in the target values.
3. Choose the **split that best reduces the variation (error)**.
4. **Repeat** the process for each branch until:

   * You reach **pure or nearly pure** groups, or
   * You hit **stopping rules** (like max depth, min samples, etc.).
5. The **leaf node’s prediction** = **average of target values** in that node.

---

### 🧮 **What It’s Trying to Minimize**

At each split, the tree tries to minimize the **Mean Squared Error (MSE)** —
i.e., make sure each group’s predicted value is as close as possible to the real values.

So it prefers splits that make data points within a node **more similar** to each other.

---

### 🌼 **Example — Predicting House Prices**

| House | Size (sqft) | Price ($) |
| ----- | ----------- | --------- |
| A     | 1000        | 100k      |
| B     | 1200        | 120k      |
| C     | 3000        | 350k      |
| D     | 3200        | 360k      |

The Decision Tree might split like this:

```
Is Size < 2000?
 ├── Yes → Avg(100k, 120k) = 110k
 └── No  → Avg(350k, 360k) = 355k
```

So:

* For small houses → predicts **$110k**
* For large houses → predicts **$355k**

✅ Each leaf gives an **average** prediction for its region.

---

### 🧩 **Key Terms**

| Term                | Meaning                                     |
| ------------------- | ------------------------------------------- |
| **Node**            | Decision point or split based on a feature  |
| **Branch**          | Outcome of a condition (Yes/No, True/False) |
| **Leaf node**       | Final group that outputs a prediction       |
| **Depth**           | How many decisions from root to leaf        |
| **Split Criterion** | How to decide the “best” split (MSE, MAE)   |

---

### ⚙️ **Common Hyperparameters**

| Parameter           | Meaning                                                    |
| ------------------- | ---------------------------------------------------------- |
| `max_depth`         | Maximum levels in tree (prevents overfitting)              |
| `min_samples_split` | Minimum samples required to split a node                   |
| `min_samples_leaf`  | Minimum samples required at a leaf                         |
| `max_features`      | Number of features to consider when looking for best split |

---

### 💡 **Advantages**

* Easy to **understand** and **visualize**
* No need for **scaling or normalization**
* Handles **non-linear** data patterns easily
* Works with **mixed types** of features

---

### ⚠️ **Limitations**

* **Overfits easily** if the tree is deep (captures noise)
* **Unstable** — small data changes can change the tree
* **Not smooth** — predictions jump suddenly between leaf regions

---

### 📈 **Key Takeaway**

> A **Decision Tree Regressor** splits data into smaller and smaller regions
> so that each region contains points with similar target values.
> The prediction in each region = **average** of those target values.

It’s like dividing a city map into neighborhoods,
then predicting house price based on **average prices** in that neighborhood.

