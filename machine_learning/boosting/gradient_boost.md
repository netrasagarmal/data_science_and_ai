# Gradient Boosting & XGBoost

## Gradient Boosting

The most important idea to lock in is this:

> **Gradient Boosting does not build Tree 2 to predict `y` again. Tree 2 is trained to correct the mistakes that the current ensemble is making.**

For regression with squared error, those “mistakes” are literally the residuals:

$$ r_i = 
y_i -\hat y_i 
$$

For other loss functions, they are more generally the **negative gradients of the loss**, so we call them **pseudo-residuals**.

---

# 1. First understand the big picture

Suppose we have:

```text
X = experience, degree
y = salary
```

Gradient Boosting builds:

```text
Base prediction
      ↓
Tree 1 learns errors
      ↓
Updated prediction
      ↓
Tree 2 learns remaining errors
      ↓
Updated prediction
      ↓
Tree 3 learns remaining errors
      ↓
...
```

Mathematically:

$$
F_0(x) = \text{initial/base prediction}
$$

Then:

$$
F_1(x)=F_0(x)+\eta h_1(x)
$$

$$
F_2(x)=F_1(x)+\eta h_2(x)
$$

and generally:

$$
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}h_m(x)
$$

where:

- $F_M(x)$ = final model
- $h_m(x)$ = decision tree $m$
- $\eta$ = learning rate

The critical thing is:

> **Each tree is trained using the same input features X, but its target is changed from original `y` to the current errors/gradients.**

---

# 2. Why do we need a base model?

Imagine these salaries:

```text
45, 50, 58, 65, 70, 82, 88, 95, 78, 110
```

Before knowing anything about experience or degree, what is the best single salary we can predict?

For squared error:

$$
F_0=\frac{\sum y_i}{n}
$$

Here:

$$
F_0=74.1
$$

So initially:

```text
Every person → predicted salary = 74.1
```

Why mean?

Because the constant that minimizes:

$$
\sum_i(y_i-c)^2
$$

is the mean.

So our starting model is:

$$
F_0(x)=74.1
$$

This is the **base model**.

Notice something important:

### The base model doesn't have to be a decision tree.

It can simply be a constant.

---

# 3. Now Tree 1

Here is our dataset.

| Row | Experience | Degree | Actual salary |
|---:|---:|---|---:|
| 1 | 1 | UG | 45 |
| 2 | 2 | UG | 50 |
| 3 | 3 | PG | 58 |
| 4 | 4 | PG | 65 |
| 5 | 5 | PG | 70 |
| 6 | 6 | PhD | 82 |
| 7 | 7 | PhD | 88 |
| 8 | 8 | PhD | 95 |
| 9 | 9 | PG | 78 |
| 10 | 10 | PhD | 110 |

We start with:

$$
\hat y=74.1
$$

for everyone.

Now calculate the residual:

$$
r_i=y_i-\hat y_i
$$

| Row | Exp | Degree | Actual $y$ | Base prediction | Residual $y-\hat y$ |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | UG | 45 | 74.1 | **-29.1** |
| 2 | 2 | UG | 50 | 74.1 | **-24.1** |
| 3 | 3 | PG | 58 | 74.1 | **-16.1** |
| 4 | 4 | PG | 65 | 74.1 | **-9.1** |
| 5 | 5 | PG | 70 | 74.1 | **-4.1** |
| 6 | 6 | PhD | 82 | 74.1 | **7.9** |
| 7 | 7 | PhD | 88 | 74.1 | **13.9** |
| 8 | 8 | PhD | 95 | 74.1 | **20.9** |
| 9 | 9 | PG | 78 | 74.1 | **3.9** |
| 10 | 10 | PhD | 110 | 74.1 | **35.9** |

Now comes the important part.

---

# 4. What exactly does Tree 1 learn?

We give the decision tree:

```text
X:
    experience
    degree

Target:
    residual
```

NOT:

```text
Target = salary
```

The tree sees:

```text
experience, degree → residual
```

For example:

```text
Experience = 1, Degree = UG → residual = -29.1
Experience = 2, Degree = UG → residual = -24.1
...
```

The tree tries to find splits that group similar residuals.

For our example, suppose it learns approximately:

```text
                 Experience <= 5.5?
                  /            \
                Yes             No
                /                \
       Experience <= 2.5?    Experience <= 9.5?
          /       \             /        \
       -26.60    -9.77       11.65      35.90
```

Those leaf values are the **average residuals of the training samples reaching that leaf**.

For example, first leaf:

Rows 1 and 2:

$$
\frac{-29.1+(-24.1)}{2}
=-26.6
$$

So Tree 1 says:

> "For people in this region of feature space, the current model is approximately **26.6 too high**."

That is what the tree is learning.

---

# 5. This is where learning rate comes in

Suppose:

$$
\eta=0.5
$$

Tree 1 predicts:

$$
h_1(x)
$$

But we don't add the entire tree prediction.

We add:

$$
0.5h_1(x)
$$

So:

$$
F_1(x)=F_0(x)+0.5h_1(x)
$$

This prevents one tree from making an excessively large correction.

---

# 6. Complete Tree 1 calculation

| Row | Actual y | Base | Residual | Tree 1 output | $0.5\times T_1$ | New prediction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45 | 74.10 | -29.10 | -26.60 | -13.30 | **60.80** |
| 2 | 50 | 74.10 | -24.10 | -26.60 | -13.30 | **60.80** |
| 3 | 58 | 74.10 | -16.10 | -9.77 | -4.88 | **69.22** |
| 4 | 65 | 74.10 | -9.10 | -9.77 | -4.88 | **69.22** |
| 5 | 70 | 74.10 | -4.10 | -9.77 | -4.88 | **69.22** |
| 6 | 82 | 74.10 | 7.90 | 11.65 | 5.83 | **79.93** |
| 7 | 88 | 74.10 | 13.90 | 11.65 | 5.83 | **79.93** |
| 8 | 95 | 74.10 | 20.90 | 11.65 | 5.83 | **79.93** |
| 9 | 78 | 74.10 | 3.90 | 11.65 | 5.83 | **79.93** |
| 10 | 110 | 74.10 | 35.90 | 35.90 | 17.95 | **92.05** |

So now we have a much better model.

The prediction changed from:

```text
74.1 for everybody
```

to something based on experience.

---

# 7. Now Tree 2

This is the part people often misunderstand.

We **do not** calculate:

```text
Actual salary - Tree 1 output
```

Instead:

> We calculate the error of the **entire current ensemble**.

So:

$$
r^{(2)}_i=y_i-F_1(x_i)
$$

For example, Row 1:

$$
45-60.8=-15.8
$$

Row 10:

$$
110-92.05=17.95
$$

So Tree 2 receives these new residuals.

Using the actual calculations:

| Row | Actual | Prediction after T1 | New residual |
|---:|---:|---:|---:|
| 1 | 45 | 60.80 | **-15.80** |
| 2 | 50 | 60.80 | **-10.80** |
| 3 | 58 | 69.22 | **-11.22** |
| 4 | 65 | 69.22 | **-4.22** |
| 5 | 70 | 69.22 | **0.78** |
| 6 | 82 | 79.93 | **2.08** |
| 7 | 88 | 79.93 | **8.08** |
| 8 | 95 | 79.93 | **15.08** |
| 9 | 78 | 79.93 | **-1.93** |
| 10 | 110 | 92.05 | **17.95** |

Now Tree 2 learns:

```text
X → new residual
```

It might discover that **degree** now helps explain some of the remaining errors.

For our example:

```text
                  Degree <= 1.5?
                  /             \
                Yes              No
                /                 \
       Experience <= 3.5?    Experience <= 7.5?
          /       \              /        \
       -12.61    -1.79         5.08       16.51
```

Again, those numbers are residual corrections.

---

# 8. Final prediction after Tree 2

Now:

$$
F_2(x)=F_1(x)+0.5T_2(x)
$$

| Row | Actual | After T1 | T2 output | 0.5 × T2 | **Final prediction** |
|---:|---:|---:|---:|---:|---:|
| 1 | 45 | 60.80 | -12.61 | -6.30 | **54.50** |
| 2 | 50 | 60.80 | -12.61 | -6.30 | **54.50** |
| 3 | 58 | 69.22 | -12.61 | -6.30 | **62.91** |
| 4 | 65 | 69.22 | -1.79 | -0.89 | **68.32** |
| 5 | 70 | 69.22 | -1.79 | -0.89 | **68.32** |
| 6 | 82 | 79.93 | 5.08 | 2.54 | **82.46** |
| 7 | 88 | 79.93 | 5.08 | 2.54 | **82.46** |
| 8 | 95 | 79.93 | 16.51 | 8.26 | **88.18** |
| 9 | 78 | 79.93 | -1.79 | -0.89 | **79.03** |
| 10 | 110 | 92.05 | 16.51 | 8.26 | **100.31** |

So our final model is:

$$
\boxed{
F_2(x)=74.1+0.5T_1(x)+0.5T_2(x)
}
$$

That's gradient boosting in its simplest form.

---

# 9. Notice what happened to the loss

Using MSE:

$$
MSE=\frac{1}{n}\sum_i(y_i-\hat y_i)^2
$$

Initially:

$$
MSE=378.29
$$

After Tree 1:

$$
MSE\approx113.32
$$

After Tree 2:

$$
MSE\approx32.08
$$

So:

```text
Base model
MSE = 378.29
       ↓
Tree 1
MSE = 113.32
       ↓
Tree 2
MSE = 32.08
```

Every new tree is trying to move the model toward lower loss.

---

# 10. But why are residuals connected to optimization?

This is the deeper concept you should understand for interviews.

Our loss is:

$$
L(y,F(x))=\frac12(y-F(x))^2
$$

Take derivative with respect to prediction $F(x)$:

$$
\frac{\partial L}{\partial F} =
F(x)-y
$$

The **negative gradient** is:

$$
-\frac{\partial L}{\partial F}=
y-F(x)
$$

And:

$$
y-F(x) = \text{residual}
$$

Therefore:

> **For squared-error regression, residuals are exactly the negative gradients of the loss.**

That's why gradient boosting is called **gradient** boosting.

---

# 11. This gives us the general algorithm

At iteration $m$:

### Step 1 — Current prediction

$$
F_{m-1}(x)
$$

### Step 2 — Calculate loss gradient

$$
r_{im} =
-\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}
$$

### Step 3 — Train a decision tree

Train:

```text
X → pseudo-residual
```

to obtain:

$$
h_m(x)
$$

### Step 4 — Add a small correction

$$
F_m(x)=F_{m-1}(x)+\eta h_m(x)
$$

### Step 5 — Repeat.

---

# 12. Important: Gradient Boosting doesn't normally "update weights" like AdaBoost

This distinction is **very important**.

You may hear:

> "Boosting gives more weight to incorrectly predicted samples."

That's particularly associated with **AdaBoost**.

In gradient boosting:

```text
AdaBoost
→ explicitly updates sample weights

Gradient Boosting
→ calculates gradients/pseudo-residuals
→ fits trees to those gradients
→ updates the model/function
```

So for your interview, don't casually say:

> "Gradient boosting increases the weight of incorrectly predicted rows."

Better:

> "Gradient boosting doesn't necessarily reweight samples explicitly like AdaBoost. It computes the negative gradient of the loss for each observation and trains the next tree to approximate those gradients."

That's a much more accurate explanation.

---

# 13. What exactly is the decision tree optimizing?

This is another subtle point.

The tree itself is still a **decision tree**.

It is trying to partition the data into leaves where the target values are similar.

But the target is now:

```text
residual / pseudo-residual
```

For squared-error boosting:

```text
Original target:
salary

Tree 1 target:
y - base_prediction

Tree 2 target:
y - current_prediction

Tree 3 target:
y - current_prediction
```

Therefore:

> The tree's job is to discover **regions of feature space where the current model needs a similar correction**.

This is probably the best mental model.

---

# 14. Think about one new person

Suppose:

```text
Experience = 8
Degree = PhD
```

Our model doesn't say:

```text
Tree 1 → salary = 95
Tree 2 → salary = 110
```

Instead:

```text
Base:
74.1

Tree 1:
+11.65

learning rate:
× 0.5

Tree 2:
+16.51

learning rate:
× 0.5
```

Therefore:

$$
74.1+(0.5)(11.65)+(0.5)(16.51)
$$

$$
=88.18
$$

The trees are **correction functions**.

---

# 15. Why multiple small trees instead of one giant tree?

Imagine using one huge decision tree.

It could potentially learn:

```text
Experience + Degree → Salary
```

very aggressively and memorize the training data.

Gradient boosting instead says:

> "I'll make many small corrections."

For example:

```text
Base model
74.1
   ↓
+ small correction
   ↓
+ another correction
   ↓
+ another correction
   ↓
+ another correction
   ↓
...
```

This makes the model more controllable and allows the ensemble to progressively reduce the loss.

That's where:

- tree depth
- number of trees
- learning rate
- regularization
- subsampling

become important.

---

# 16. Learning rate vs number of trees

Suppose:

$$
\eta=1
$$

A tree's full correction is added.

If:

$$
\eta=0.1
$$

only 10% of the correction is added.

For example:

```text
Tree predicts residual = +20
```

With:

```text
learning rate = 1
→ +20

learning rate = 0.5
→ +10

learning rate = 0.1
→ +2
```

Usually:

```text
smaller learning rate
       +
more trees
```

is a common strategy.

The trade-off is:

```text
small η → slower learning, often better controlled
large η → faster learning, potentially overfit more easily
```

---

# 17. Now the important classification case: Iris

This is where things become slightly different.

Your Iris target is:

```text
0 = setosa
1 = versicolor
2 = virginica
```

This is **multiclass classification**.

You cannot simply do:

$$
residual=y-\hat y
$$

because `y` isn't a continuous salary.

Instead, gradient boosting can use **log loss / cross-entropy**.

For each observation we have probabilities:

$$
P(setosa),P(versicolor),P(virginica)
$$

and the model produces class scores/logits:

$$
F_0(x),F_1(x),F_2(x)
$$

Then softmax converts those scores into probabilities:

$$
P_k=
\frac{e^{F_k}}
{\sum_j e^{F_j}}
$$

---

# 18. Important difference: multiclass boosting uses multiple trees

This is an important correction to your "2 decision trees" request.

For 3-class Iris, a standard multiclass gradient boosting implementation generally builds:

> **one tree per class per boosting iteration.**

Therefore:

```text
Boosting iteration 1
    Tree 1 → setosa correction
    Tree 2 → versicolor correction
    Tree 3 → virginica correction

Boosting iteration 2
    Tree 4 → setosa correction
    Tree 5 → versicolor correction
    Tree 6 → virginica correction
```

So if you want **2 boosting iterations**, that's typically **6 class-specific trees**, not 2 total trees.

That is the correct way to understand the multiclass case.

---

# 19. Tiny Iris example

Let's use six Iris observations:

| Row | Sepal L | Sepal W | Petal L | Petal W | Actual |
|---:|---:|---:|---:|---:|---|
| 1 | 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
| 2 | 4.9 | 3.0 | 1.4 | 0.2 | Setosa |
| 3 | 7.0 | 3.2 | 4.7 | 1.4 | Versicolor |
| 4 | 6.4 | 3.2 | 4.5 | 1.5 | Versicolor |
| 5 | 6.3 | 3.3 | 6.0 | 2.5 | Virginica |
| 6 | 5.8 | 2.7 | 5.1 | 1.9 | Virginica |

We have three classes:

$$
K=3
$$

---

# 20. Base model for Iris

We initially know nothing.

Because there are equal numbers of each class:

```text
Setosa = 2/6
Versicolor = 2/6
Virginica = 2/6
```

So:

$$
P=(1/3,1/3,1/3)
$$

For every observation:

| Row | Setosa P | Versicolor P | Virginica P |
|---:|---:|---:|---:|
| 1 | .333 | .333 | .333 |
| 2 | .333 | .333 | .333 |
| 3 | .333 | .333 | .333 |
| 4 | .333 | .333 | .333 |
| 5 | .333 | .333 | .333 |
| 6 | .333 | .333 | .333 |

---

# 21. Now calculate pseudo-residuals

For classification with cross-entropy, conceptually:

$$
r_{ik}=y_{ik}-p_{ik}
$$

where:

$$
y_{ik}=
\begin{cases}
1 & \text{if observation belongs to class } k\\
0 & \text{otherwise}
\end{cases}
$$

For Row 1:

Actual = Setosa.

Therefore:

```text
y = [1, 0, 0]
p = [0.333, 0.333, 0.333]
```

So:

$$
r=[0.667,-0.333,-0.333]
$$

For Row 3:

Actual = Versicolor.

$$
y=[0,1,0]
$$

Therefore:

$$
r=[-0.333,0.667,-0.333]
$$

And Row 5:

$$
r=[-0.333,-0.333,0.667]
$$

Complete table:

| Row | Actual | r(Setosa) | r(Versicolor) | r(Virginica) |
|---:|---|---:|---:|---:|
| 1 | Setosa | **0.667** | -0.333 | -0.333 |
| 2 | Setosa | **0.667** | -0.333 | -0.333 |
| 3 | Versicolor | -0.333 | **0.667** | -0.333 |
| 4 | Versicolor | -0.333 | **0.667** | -0.333 |
| 5 | Virginica | -0.333 | -0.333 | **0.667** |
| 6 | Virginica | -0.333 | -0.333 | **0.667** |

Now we train **three regression trees**.

---

# 22. Tree for Setosa

Tree 1's target is:

```text
Setosa residual
```

The tree might discover:

```text
Petal/sepal measurements
          ↓
      Setosa?
```

For our tiny dataset, a simple split such as:

```text
Sepal length <= 5.45
```

can separate the two Setosa examples.

The tree therefore predicts approximately:

```text
Setosa tree:

Setosa region      +0.667
Other region       -0.333
```

Similarly:

### Versicolor tree

```text
Sepal length <= 6.35
       ↓
Versicolor correction
```

### Virginica tree

```text
Petal length > 4.9
       ↓
Virginica correction
```

Again, these are **corrections to class scores**, not final probabilities.

---

# 23. Apply learning rate

Let's use:

$$
\eta=0.5
$$

For Row 1:

```text
Setosa tree       +0.667
Versicolor tree   -0.333
Virginica tree    -0.333
```

Multiply by 0.5:

```text
+0.333
-0.167
-0.167
```

Our initial class scores were:

```text
[0, 0, 0]
```

So new scores become:

$$
F=[0.333,-0.167,-0.167]
$$

These are **not probabilities yet**.

---

# 24. Convert scores into probabilities

Use softmax:

$$
P_k=
\frac{e^{F_k}}
{\sum_j e^{F_j}}
$$

For Row 1:

```text
Scores:

Setosa       0.333
Versicolor  -0.167
Virginica   -0.167
```

Softmax gives approximately:

```text
Setosa       0.452
Versicolor   0.274
Virginica    0.274
```

So prediction:

```text
Setosa
```

because:

$$
0.452 > 0.274
$$

---

# 25. After the first boosting iteration

Our six rows become approximately:

| Row | Actual | Setosa P | Versicolor P | Virginica P | Prediction |
|---:|---|---:|---:|---:|---|
| 1 | Setosa | **0.452** | .274 | .274 | Setosa |
| 2 | Setosa | **0.452** | .274 | .274 | Setosa |
| 3 | Versicolor | .274 | **.452** | .274 | Versicolor |
| 4 | Versicolor | .274 | **.452** | .274 | Versicolor |
| 5 | Virginica | .274 | .274 | **.452** | Virginica |
| 6 | Virginica | .274 | .274 | **.452** | Virginica |

Notice:

The model has become more confident.

Initially:

```text
33.3 / 33.3 / 33.3
```

After Tree round 1:

```text
45.2 / 27.4 / 27.4
```

---

# 26. Then the second boosting iteration

Now we **do not go back to the original residuals**.

We calculate new probabilities:

```text
P = [0.452, 0.274, 0.274]
```

Then calculate new pseudo-residuals:

$$
r=y-p
$$

For a Setosa row:

$$
r=[1-.452,\;0-.274,\;0-.274]
$$

$$
r=[0.548,-0.274,-0.274]
$$

For a Versicolor row:

$$
r=[-0.274,0.548,-0.274]
$$

For a Virginica row:

$$
r=[-0.274,-0.274,0.548]
$$

So the new trees now have smaller corrections because the model is already moving in the right direction.

---

# 27. Second round

The three trees produce approximately:

```text
Setosa tree:
    Setosa region       +0.548
    other regions      -0.274

Versicolor tree:
    Versicolor region  +0.548
    other regions      -0.274

Virginica tree:
    Virginica region   +0.548
    other regions      -0.274
```

After learning rate:

$$
0.5\times0.548=0.274
$$

So for a Setosa example, the score becomes:

$$
F_{setosa}=0.333+0.274=0.607
$$

and the other two:

$$
F=-0.167-0.137=-0.304
$$

So:

```text
Scores:

Setosa       0.607
Versicolor  -0.304
Virginica   -0.304
```

Softmax gives approximately:

```text
Setosa       0.554
Versicolor   0.223
Virginica    0.223
```

Again:

$$
\boxed{\text{Setosa}}
$$

---

# 28. Final Iris example

After two boosting iterations:

| Row | Actual | Setosa | Versicolor | Virginica | Final class |
|---:|---|---:|---:|---:|---|
| 1 | Setosa | **0.554** | .223 | .223 | **Setosa** |
| 2 | Setosa | **0.554** | .223 | .223 | **Setosa** |
| 3 | Versicolor | .223 | **.554** | .223 | **Versicolor** |
| 4 | Versicolor | .223 | **.554** | .223 | **Versicolor** |
| 5 | Virginica | .223 | .223 | **.554** | **Virginica** |
| 6 | Virginica | .223 | .223 | **.554** | **Virginica** |

The classification decision is simply:

$$
\boxed{\arg\max_k P_k}
$$

So if:

```text
Setosa       = 0.223
Versicolor   = 0.554
Virginica    = 0.223
```

then:

```text
prediction = Versicolor
```

---

# 29. Salary vs Iris — put them side by side

This is the connection I want you to remember.

| Concept | Salary regression | Iris classification |
|---|---|---|
| Target | Continuous salary | Class |
| Loss | MSE | Cross-entropy/log loss |
| Base model | Mean salary | Initial class scores/prior probabilities |
| Tree target | Residual | Pseudo-residual / negative gradient |
| Tree type | Regression tree | Regression trees on gradients |
| Output | Correction to salary | Correction to class score |
| Learning rate | Shrinks correction | Shrinks correction |
| Final output | Numeric salary | Class probabilities |
| Final decision | Direct numeric value | `argmax(probabilities)` |

---

# 30. The most important mental model

Don't think:

```text
Tree 1 predicts salary
Tree 2 predicts salary
Tree 3 predicts salary
```

Instead think:

```text
Base model
    ↓
"What's wrong?"
    ↓
Tree 1 learns correction
    ↓
Updated model
    ↓
"What's still wrong?"
    ↓
Tree 2 learns correction
    ↓
Updated model
    ↓
"What's still wrong?"
    ↓
Tree 3 learns correction
    ↓
...
```

For regression:

```text
Residual = actual - current prediction
```

For general gradient boosting:

```text
Pseudo-residual = negative gradient of loss
```

For classification:

```text
current probabilities
       ↓
negative gradient
       ↓
class-specific trees
       ↓
class scores
       ↓
softmax
       ↓
probabilities
       ↓
argmax
       ↓
predicted class
```

---

# 31. One subtle but VERY important point

Suppose the original salary is:

```text
y = 110
```

and after Tree 1:

```text
prediction = 92.05
```

The residual is:

$$
110-92.05=17.95
$$

Tree 2 doesn't know:

> "This person's salary is 110."

It is effectively being told:

> "For this person's feature region, the current model needs approximately **+17.95 more prediction**."

And the tree learns patterns across **all rows** to figure out:

> "Which combinations of experience/degree tend to need positive or negative corrections?"

That is why the **same input features X** are used again, but the **target given to the tree changes at every iteration**.

---

# 32. What should you say in an interview?

A strong experienced-level answer would be:

> **"Gradient boosting starts with a base model, typically the mean for squared-error regression. At every iteration, we calculate the negative gradient of the loss with respect to the current predictions; for MSE this is simply the residual $y-\hat y$. We then train a shallow decision tree on the original features but use these pseudo-residuals as the target, so the tree learns where and how the current ensemble needs correction. Its output is scaled by the learning rate and added to the existing model, and we repeat this process until the desired number of trees or stopping criteria is reached."**

And for classification:

> **"For multiclass classification, the model maintains class scores, converts them to probabilities using softmax, computes class-wise negative gradients from cross-entropy, and typically fits one regression tree per class at each boosting iteration. The trees update the class scores, which are converted back to probabilities, and the class with the highest final probability becomes the prediction."**

---

## The 5 formulas you should memorize

### 1. Base regression model

$$
\boxed{F_0=\arg\min_c\sum_i L(y_i,c)}
$$

For MSE:

$$
\boxed{F_0=\text{mean}(y)}
$$

### 2. Residual / negative gradient

$$
\boxed{
r_{im}=
-\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
}
$$

For MSE:

$$
\boxed{r=y-F(x)}
$$

### 3. Tree fitting

$$
\boxed{h_m(x)\approx r_m(x)}
$$

### 4. Model update

$$
\boxed{
F_m(x)=F_{m-1}(x)+\eta h_m(x)
}
$$

### 5. Multiclass prediction

$$
\boxed{
P_k=\frac{e^{F_k}}{\sum_j e^{F_j}}
}
$$

and:

$$
\boxed{
\hat y=\arg\max_k P_k
}
$$

**If you understand those five equations and the salary table above, you understand the core mechanics of Gradient Boosting rather than just memorizing the definition.**


---
