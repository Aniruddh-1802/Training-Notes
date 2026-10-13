# Machine Learning Fundamentals

**Date:** 03/08/2026

---

# Keywords

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Rule-based Systems
- Automation
- Neural Networks
- Arthur Samuel
- Model
- Hypothesis
- Feature
- Feature Vector
- Target
- Label
- Supervised Learning
- Unsupervised Learning
- Labeled Data
- Unlabeled Data
- Classification
- Binary Classification
- Multi-Class Classification
- Regression

---

# Tags / Keywords Used

## Abbreviations Used

| Tag | Meaning |
|------|---------|
| LIB | Library |
| USA | Usage (Example Code) |
| PAR | Parameters |
| SYN | Syntax |
| WHY | Why this is used |
| EXA | Example |
| MET | Methods |
| CAT | Cannot have it |
| ALI | Alias |
| RES | Expected Result |
| OPT | Optional |
| ANE | Another Example |
| FUN | Functions Used |
| DAM | Data Members |
| TYS | Types |
| VER | Python Version |
| USC | Use Cases |
| COM | CLI Command |
| INS | Install this Library |
| STE | Steps |

---

# Table of Contents

- Introduction
- Learning Hierarchy
- Machine Learning
- Core Terminology
  - Model
  - Feature
  - Target (Label)
- Types of Machine Learning
  - Supervised Learning
    - Labeled Data
    - Classification
      - Binary Classification
      - Multi-Class Classification
    - Regression
  - Unsupervised Learning
    - Unlabeled Data

---

# Introduction

Artificial Intelligence is a broad field focused on enabling computers to perform tasks that normally require human intelligence.

The learning hierarchy can be represented as follows:

```
Artificial Intelligence
        │
        ├── Neural Networks
              │
              └── Deep Learning
                    │
                    └── Machine Learning
```

## Rule-based Systems

Rule-based systems follow predefined logical rules.

```
Rule-based Systems = Automation
```

These systems do **not** learn from data. Every condition must be explicitly programmed.

> **Added Context:** Rule-based automation executes predefined IF-THEN rules and does not improve automatically from experience or new data.

---

# Machine Learning

Machine Learning is a subset of Artificial Intelligence that enables systems to learn patterns from data instead of relying solely on manually written rules.

## Definition

Arthur Samuel, a pioneer in the field of Artificial Intelligence and computer gaming, coined the term **Machine Learning** as:

> "Field of study that gives computers the capability to learn without being explicitly programmed."

---

# Core Terminology

## Model

A **Model** is a specific representation learned from data by applying a Machine Learning algorithm.

A model is also called a **Hypothesis**.

> **Added Context:** After training, the model is used to make predictions on unseen data.

---

## Feature

A **Feature** is an individual measurable property of the data.

A collection of numerical features can be represented as a **Feature Vector**.

Feature vectors are provided as input to the Machine Learning model.

### Example

For a fruit prediction dataset:

| Feature | Value |
|----------|------|
| Color | Red |
| Weight | 180 g |
| Diameter | 8 cm |

These values collectively form the feature vector.

---

## Target (Label)

A **Target Variable** (or **Label**) is the value that the Machine Learning model is expected to predict.

For the fruit example:

| Features | Target(Label) |
|-----------|---------------|
| Red, 180g, 8cm | Apple |
| Orange, 150g, 7cm | Orange |
| Yellow, 120g | Banana |

---

# Types of Machine Learning

[TYS]

Machine Learning is broadly classified into:

1. Supervised Learning
2. Unsupervised Learning

---

# Supervised Learning

Supervised Learning learns from **Labeled Data** to predict outputs.

## Labeled Data

Labeled Data consists of known input-output pairs.

Example:

| Input | Output |
|--------|--------|
| Fruit Features | Apple |
| Fruit Features | Orange |
| Fruit Features | Banana |

Since the correct answers are already known during training, the algorithm learns the relationship between inputs and outputs.

---

## Types of Supervised Learning

[TYS]

### Classification

Classification predicts **discrete categories or classes**.

Examples:

- Spam / Not Spam
- Fraud / Not Fraud
- Apple / Orange / Banana

---

### Types of Classification

[TYS]

#### Binary Classification

Binary Classification predicts one of **two possible classes**.

Examples

- Yes / No
- True / False
- Pass / Fail
- Spam / Not Spam

---

#### Multi-Class Classification

Multi-Class Classification predicts one class from **three or more possible classes**.

Examples

- Apple
- Orange
- Banana
- Mango

---

### Regression

Regression predicts **continuous numerical values** instead of categories.

Examples

- House Price
- Temperature
- Salary Prediction
- Stock Price

---

# Unsupervised Learning

Unsupervised Learning learns patterns from **Unlabeled Data**.

Unlike Supervised Learning, the correct outputs are not available during training.

> **Added Context:** The algorithm attempts to discover hidden structures, similarities, or groupings within the data.

---

## Unlabeled Data

Unlabeled Data consists of input data without known target labels.

Example

| Features | Label |
|-----------|-------|
| Red, 180g | Unknown |
| Orange, 150g | Unknown |
| Yellow, 120g | Unknown |

The algorithm must identify patterns without knowing the correct answers.

---

# Summary

```
Artificial Intelligence
│
├── Rule-based Systems
│     └── Automation
│
└── Machine Learning
      │
      ├── Supervised Learning
      │      ├── Classification
      │      │      ├── Binary Classification
      │      │      └── Multi-Class Classification
      │      │
      │      └── Regression
      │
      └── Unsupervised Learning
```

---

# Important Definitions

| Term | Definition |
|------|------------|
| Artificial Intelligence | Field of building intelligent systems capable of performing tasks that typically require human intelligence. |
| Machine Learning | Field of study that gives computers the capability to learn without being explicitly programmed. |
| Model | Learned representation produced by a Machine Learning algorithm; also called a hypothesis. |
| Feature | An individual measurable property used as input to a model. |
| Feature Vector | Collection of multiple features representing one data sample. |
| Target (Label) | Desired output that the model predicts. |
| Labeled Data | Dataset containing both inputs and their correct outputs. |
| Unlabeled Data | Dataset containing inputs without corresponding target outputs. |
| Classification | Predicts discrete categories. |
| Regression | Predicts continuous numerical values. |

# Types of Machine Learning

Machine Learning can be categorized into several learning paradigms depending on how data is provided and how the model learns from it.

```
Machine Learning
│
├── 1. Supervised Learning
│   └── Uses labeled data
│
├── 2. Unsupervised Learning
│   └── Uses unlabeled data
│
├── 3. Semi-Supervised Learning
│   └── Uses both labeled and unlabeled data
│
├── 4. Reinforcement Learning
│   └── Learns using rewards and penalties
│
├── 5. Self-Supervised Learning
│   └── Creates labels from data itself
│
├── 6. Active Learning
│   └── Model selects important data for labeling
│
├── 7. Online Learning
│   └── Learns continuously from incoming data
│
├── 8. Transfer Learning
│   └── Reuses knowledge from a trained model
│
├── 9. Ensemble Learning
│   └── Combines multiple models for better accuracy
│
├── 10. Federated Learning
│   └── Trains models across devices without sharing data
│
├── 11. Multi-Task Learning
│   └── Learns multiple tasks using one model
│
├── 12. Meta Learning
│   └── Learns how to learn efficiently
│
├── 13. Incremental (Continual) Learning
│   └── Learns continuously without forgetting old knowledge
│
└── 14. Weakly Supervised Learning
    └── Uses limited or inaccurate labels
```

---

## 1. Supervised Learning

Uses **labeled data** to learn a mapping between inputs and outputs.

> **Added Context:** The model is trained using input-output pairs where the correct target (label) is already known.

---

## 2. Unsupervised Learning

Uses **unlabeled data** to discover hidden patterns, relationships, or structures within the data.

> **Added Context:** Since no target labels are available, the algorithm identifies patterns such as clusters or feature relationships.

---

## 3. Semi-Supervised Learning

Uses **both labeled and unlabeled data** during training.

> **Added Context:** Typically, a small amount of labeled data and a large amount of unlabeled data are combined to improve model performance while reducing labeling costs.

---

## 4. Reinforcement Learning

Learns by interacting with an environment using **rewards and penalties**.

> **Added Context:** The objective is to maximize the cumulative reward by learning the best sequence of actions.

---

## 5. Self-Supervised Learning

Creates labels **automatically from the data itself**.

> **Added Context:** The model generates its own supervision by predicting one part of the data from another. This approach is widely used in modern Large Language Models (LLMs) and computer vision.

---

## 6. Active Learning

The model selects the **most informative data samples** to be labeled by humans.

> **Added Context:** Active Learning reduces labeling effort by requesting annotations only for data points that will most improve the model.

---

## 7. Online Learning

Learns **continuously from incoming data** instead of training once on a fixed dataset.

> **Added Context:** Suitable for applications where new data arrives continuously, such as recommendation systems and fraud detection.

---

## 8. Transfer Learning

Reuses knowledge from a **previously trained model** for a new but related task.

> **Added Context:** Instead of training from scratch, an existing model is fine-tuned, significantly reducing training time and data requirements.

---

## 9. Ensemble Learning

Combines predictions from **multiple machine learning models** to improve overall performance.

> **Added Context:** Common ensemble methods include Bagging, Boosting, and Stacking.

---

## 10. Federated Learning

Trains models across **multiple devices or servers** without sharing the original data.

> **Added Context:** Only model updates are shared, improving user privacy and reducing centralized data collection.

---

## 11. Multi-Task Learning

Uses **one model** to learn multiple related tasks simultaneously.

> **Added Context:** Learning related tasks together allows the model to share useful representations and improve generalization.

---

## 12. Meta Learning

Learns **how to learn** efficiently.

> **Added Context:** Often referred to as "Learning to Learn," Meta Learning enables models to adapt quickly to new tasks using only a small amount of data.

---

## 13. Incremental (Continual) Learning

Learns continuously **without forgetting previously acquired knowledge**.

> **Added Context:** The primary challenge is avoiding **Catastrophic Forgetting**, where learning new information causes the model to lose previously learned knowledge.

---

## 14. Weakly Supervised Learning

Uses **limited, noisy, or inaccurate labels** for training.

> **Added Context:** This approach is useful when obtaining perfectly labeled datasets is expensive or impractical.

---

# Linear Regression

Linear Regression is a supervised Machine Learning algorithm used to predict a **continuous numerical value** based on the relationship between one independent variable and one dependent variable.

---

## Definition

Linear Regression predicts a continuous value using **one independent variable** by fitting the best possible straight line through the data.

---

## [WHY]

### Why is Linear Regression used?

- Finds the **best-fit line** to model the relationship between variables.
- Helps understand how one variable changes with another.
- Used for predicting continuous numerical values.
- Simple, interpretable, and computationally efficient.

---

## [SYN]

### Mathematical Equation

\[
y = mx + c
\]

Where:

- **y = mx + c**

---

## [PAR]

### Parameters

| Parameter | Description |
|-----------|-------------|
| **X** | Independent (Input) Variable |
| **y** | Dependent (Target) Variable |
| **m** | Slope (Coefficient) indicating how much **y** changes for a unit change in **X** |
| **c** | Intercept where the regression line crosses the Y-axis |

---

## Understanding the Equation

Suppose,

\[
y = 2x + 5
\]

If

```
x = 10
```

Then

```
y = 2 × 10 + 5
  = 25
```

So, when the input is **10**, the predicted output becomes **25**.

---

## Best-Fit Line

The objective of Linear Regression is to determine the values of **m** (slope) and **c** (intercept) that produce the line with the **minimum prediction error**.

> **Added Context:** The most common technique for finding the best-fit line is the **Ordinary Least Squares (OLS)** method, which minimizes the sum of squared errors between actual and predicted values.

---

## [EXA]

### Examples

- Predicting house price based on house area.
- Predicting salary based on years of experience.
- Predicting electricity consumption based on usage.
- Predicting crop yield based on rainfall.

---

## [USC]

### Use Cases

- Sales prediction
- Stock price trend analysis
- Weather forecasting
- Demand forecasting
- Revenue estimation
- Population growth prediction
- Energy consumption prediction
- Business analytics

---

## Advantages

- Simple and easy to understand.
- Fast to train.
- Highly interpretable.
- Works well for linear relationships.
- Forms the foundation for many advanced regression techniques.

---

## Limitations

- Assumes a linear relationship between variables.
- Sensitive to outliers.
- Cannot model complex nonlinear relationships.
- Performance decreases when assumptions are violated.

---

# Multiple Linear Regression

Multiple Linear Regression is an extension of **Linear Regression** that predicts a continuous value using **two or more independent variables**.

---

## Definition

Multiple Linear Regression models the relationship between one dependent variable and multiple independent variables by fitting the best possible linear equation.

> **Added Context:** Unlike Linear Regression, which uses a single input feature, Multiple Linear Regression considers multiple factors simultaneously to improve prediction accuracy.

---

## [WHY]

### Why is Multiple Linear Regression used?

- Improves prediction accuracy by considering multiple factors.
- Models real-world problems where multiple variables influence the output.
- Estimates the contribution of each independent variable.
- Helps understand relationships among variables.

---

## [SYN]

### Mathematical Equation

$y = b_0 + b_1x_1 + b_2x_2 + \cdots + b_nx_n$

---

## [PAR]

### Parameters

| Parameter | Description |
|-----------|-------------|
| **$x₁, x₂, ..., xₙ$** | Independent (Input) Variables |
| **y** | Dependent (Target) Variable |
| **b₀** | Intercept (Constant Term) |
| **b₁, b₂, ..., bₙ** | Regression Coefficients representing each feature's contribution |

---

## Understanding the Equation

Suppose,

HousePrice = 50000 + (3000 × Area) + (500000 × Bedrooms)

If

```
Area = 1000 sq.ft
Bedrooms = 3
```

Then

```
HousePrice

= 50000 + (3000 × 1000) + (500000 × 3)

= 45,50,000
```

---

## [EXA]

### Examples

- Predicting house price using area, number of bedrooms, and location.
- Predicting student marks using study hours, attendance, and assignments.

---

## [USC]

### Use Cases

- Real estate price prediction
- Medical risk prediction
- Business revenue forecasting
- Employee salary prediction

---

## [ANE]

### Another Examples

- House Price = f(area, bedrooms, age of house)
- Car Price = f(year, mileage, engine size)

---

## Advantages

- Handles multiple input variables.
- More realistic than Simple Linear Regression.
- Easy to interpret coefficients.
- Useful for business forecasting.

---

## Limitations

- Assumes a linear relationship.
- Sensitive to multicollinearity.
- Sensitive to outliers.
- Requires independent variables to be carefully selected.

---

> **Important Note**
>
> - Linear Regression → One input feature (**X**)
> - Multiple Linear Regression → Two or more input features (**X₁, X₂, ...**)
> - Both are **Supervised Learning** algorithms used for **Regression** problems.

---

# Polynomial Regression

Polynomial Regression predicts a continuous value by fitting a **curved (non-linear) relationship** between variables.

---

## Definition

Polynomial Regression extends Linear Regression by adding polynomial terms (x², x³, ...), allowing the model to fit curved relationships.

> **Added Context:** Although the fitted curve is non-linear, Polynomial Regression is still considered a Linear Model because it is linear with respect to its coefficients.

---

## [WHY]

### Why is Polynomial Regression used?

- Used when data follows a non-linear pattern.
- Captures curved relationships that Linear Regression cannot model accurately.
- Provides better prediction for non-linear datasets.

---

## [SYN]

### Mathematical Equation

$y = b_0 + b_1x + b_2x^2 + b_3x^3 + \cdots + b_nx^n$

---

## [PAR]

| Parameter | Description |
|-----------|-------------|
| **X** | Independent (Input) Variable |
| **y** | Dependent (Target) Variable |
| **b₀** | Intercept |
| **b₁, b₂, ..., bₙ** | Coefficients |
| **n** | Degree of the Polynomial |

---

## Understanding the Equation

Example

$y = 5 + 2x + 3x^2$

If

```
x = 2
```

Then

```
y

= 5 + (2×2) + (3×4)

= 21
```

---

## [EXA]

### Examples

- Predicting plant growth over time.
- Predicting population growth.
- Predicting car fuel efficiency based on speed.

---

## [USC]

### Use Cases

- Sales trend forecasting
- Weather and climate prediction
- Biological growth analysis
- Stock market trend modeling

---

## [ANE]

### Another Examples

- House price vs distance from city (curved relationship)
- Student performance vs study hours (diminishing returns)

---

## [CAT]

### Cannot Have It

- Not suitable for data with a purely linear relationship.
- High-degree polynomials may cause overfitting.
- Can become computationally expensive for very high polynomial degrees.

---

## Advantages

- Models curved relationships.
- More flexible than Linear Regression.
- Can fit complex trends.

---

## Limitations

- Easily overfits with higher-degree polynomials.
- Sensitive to outliers.
- Degree selection is important.

---

> **Important Note**
>
> - Linear Regression → Fits a straight line.
> - Polynomial Regression → Fits a curve.
> - Polynomial Regression is still a **Regression algorithm** under **Supervised Learning**.

---

# Ridge Regression

Ridge Regression is a Linear Regression technique that reduces overfitting using **L2 Regularization**.

---

## Definition

Ridge Regression adds a penalty on the **square of the coefficient values** to prevent overly complex models.

---

## [WHY]

### Why is Ridge Regression used?

- Prevents overfitting by penalizing large coefficient values.
- Improves model performance on unseen data.
- Handles multicollinearity between features.
- Produces more stable models.

---

## [SYN]

### Cost Function

$Cost = RSS + \lambda \sum b^2$

---

## [PAR]

| Parameter | Description |
|-----------|-------------|
| **RSS** | Residual Sum of Squares (Prediction Error) |
| **λ (Lambda)** | Regularization Parameter |
| **b** | Model Coefficients (Weights) |

---

## Understanding the Cost Function

The objective is to minimize

```
Prediction Error

+

Penalty for Large Coefficients
```

A larger λ increases the penalty, resulting in smaller coefficient values.

---

## [EXA]

### Examples

- Predicting house prices with many correlated features.
- Predicting sales using multiple marketing variables.

---

## [USC]

### Use Cases

- Financial forecasting
- Medical data analysis
- Real estate price prediction
- High-dimensional datasets

---

## [ANE]

### Another Examples

- House Price = f(area, bedrooms, age, location)
- Sales = f(TV ads, Radio ads, Newspaper ads)

---

## [CAT]

### Cannot Have It

- Does not perform feature selection.
- Coefficients become small but never exactly zero.

---

## Advantages

- Reduces overfitting.
- Handles correlated features.
- Stable predictions.
- Keeps all features.

---

## Limitations

- Does not eliminate irrelevant features.
- Selecting the optimal λ is important.

---

> **Important Note**
>
> - Ridge Regression = Linear Regression + **L2 Regularization**
> - Uses the **L2 Penalty (Squared Coefficients)**
> - Best when many features are correlated.
> - λ = 0 → Same as Linear Regression.
> - Larger λ → Smaller coefficients and less overfitting.

---

# Lasso Regression

Lasso Regression is a Linear Regression technique that reduces overfitting using **L1 Regularization**.

---

## Definition

Lasso Regression adds a penalty based on the **absolute values of the coefficients**, enabling automatic feature selection.

---

## [WHY]

### Why is Lasso Regression used?

- Prevents overfitting by penalizing large coefficient values.
- Performs automatic feature selection.
- Removes less important features by making their coefficients zero.
- Produces simpler and more interpretable models.

---

## [SYN]

### Cost Function

$Cost = RSS + \lambda \sum |b|$

---

## [PAR]

| Parameter | Description |
|-----------|-------------|
| **RSS** | Residual Sum of Squares (Prediction Error) |
| **λ (Lambda)** | Regularization Parameter |
| **b** | Model Coefficients (Weights) |

---

## Understanding the Cost Function

Increasing λ increases the penalty, causing unimportant feature coefficients to shrink toward zero.

---

## [EXA]

### Examples

- Predicting house prices with many features.
- Customer churn prediction using multiple variables.

---

## [USC]

### Use Cases

- Feature selection
- Medical diagnosis
- Financial forecasting
- High-dimensional datasets

---

## [ANE]

### Another Examples

- House Price = f(area, bedrooms, location, age)
- Sales = f(TV ads, Radio ads, Newspaper ads)

---

## [CAT]

### Cannot Have It

- May remove useful features if λ is too large.
- Not ideal when all features are equally important.

---

## Advantages

- Performs automatic feature selection.
- Produces simpler models.
- Reduces overfitting.
- Improves interpretability.

---

## Limitations

- May discard useful features.
- Performance can decrease with highly correlated variables.

---

> **Important Note**
>
> - Lasso Regression = Linear Regression + **L1 Regularization**
> - Uses the **L1 Penalty (Absolute Value of Coefficients)**
> - Can reduce some coefficients exactly to **zero**.
> - Performs automatic feature selection.
> - λ = 0 → Same as Linear Regression.
> - Larger λ → More coefficients become zero, resulting in a simpler model.

---

# RSS (Residual Sum of Squares)

## Definition

RSS (**Residual Sum of Squares**) measures the total prediction error made by a regression model.

It calculates the squared difference between the **Actual** values and the **Predicted** values.

---

## Formula

\[
RSS = \sum (Actual - Predicted)^2
\]

---

## Example

| Actual | Predicted | Error | Error² |
|---------|-----------|-------|---------|
| 100 | 90 | 10 | 100 |
| 120 | 125 | -5 | 25 |
| 140 | 130 | 10 | 100 |

```
RSS

= 100 + 25 + 100

= 225
```

> **Added Context:** A smaller RSS indicates that the model's predictions are closer to the actual values, meaning the model fits the data better.


# Elastic Net Regression

---

## Definition

**Elastic Net Regression** is a regression technique that combines **L1 (Lasso)** and **L2 (Ridge)** regularization.

> Elastic Net = Ridge Regression + Lasso Regression.

It uses both **L1 and L2 penalties**.

* **L1** helps with feature selection.
* **L2** helps handle correlated features.
* It is a best choice when features are many and highly correlated.

---

## [WHY] — Why Elastic Net Regression?

* Reduces overfitting.
* Performs feature selection.
* Handles multicollinearity between features.
* Works well when there are many correlated variables.

---

## [SYN] — Cost Function

The Elastic Net cost function can be represented as:

```text
Cost = RSS + λ₁Σ|b| + λ₂Σ(b²)
```

More explicitly:

```text
Cost = RSS + λ₁ Σ|bⱼ| + λ₂ Σ(bⱼ²)
```

Where the summation is taken over the model coefficients.

---

## [PAR] — Parameters

* **RSS** → Residual Sum of Squares (prediction error).
* **λ₁** → L1 regularization strength.
* **λ₂** → L2 regularization strength.
* **b** → Model coefficients (weights).

### Additional clarification

* A higher **λ₁** increases the effect of L1 regularization and encourages more coefficients to become zero.
* A higher **λ₂** increases the effect of L2 regularization and causes coefficients to shrink more strongly.

---

## [TYS] — Types of Regularization Used

### L1 Regularization (Lasso)

* Removes less important features by making coefficients zero.
* Helps perform feature selection.

### L2 Regularization (Ridge)

* Shrinks coefficients but keeps all features.
* Helps handle correlated features and multicollinearity.

---

## [EXA] — Examples

* Predicting house prices using many related features.
* Predicting medical outcomes using multiple patient measurements.

---

## [USC] — Use Cases

* Genomics and bioinformatics.
* Financial modeling.
* Customer behavior prediction.
* High-dimensional datasets.

---

## [CAT] — Characteristics / Considerations

* Requires tuning two regularization parameters.
* More complex than Ridge and Lasso.
* Not necessary for small datasets.

---

## [ANE] — Additional Notes / Examples

* Stock prediction using many market indicators.
* Disease prediction using hundreds of medical features.

---

## Key Notes

* Elastic Net = Ridge Regression + Lasso Regression.
* Uses both L1 and L2 penalties.
* L1 helps feature selection.
* L2 helps handle correlated features.
* Best choice when features are many and highly correlated.

---

## Core Concept

Elastic Net combines the two regularization approaches:

```text
Elastic Net
    │
    ├── L1 Regularization → Lasso
    │       └── Feature Selection
    │           └── Can make coefficients zero
    │
    └── L2 Regularization → Ridge
            └── Coefficient Shrinkage
                └── Handles Correlated Features
```

Therefore, **Elastic Net Regression is particularly useful when a dataset contains many features that are highly correlated, while feature selection is also required.**

# Decision Tree Regression

## Definition

**Decision Tree Regression** is a regression algorithm that predicts continuous values using a **tree-like structure of decision rules**.

---

## [WHY] — Why Decision Tree Regression?

* Captures non-linear relationships between features and target values.
* Does not require feature scaling.
* Easy to understand and interpret.

---

## [SYN] — Prediction

```text id="dtr81q"
Prediction = Average value of data points in a leaf node
```

The final prediction is usually the **average value of samples in each leaf node**.

---

## [PAR] — Parameters / Components

* **Root Node** → Starting point containing all data.
* **Decision Node** → Splits data based on conditions.
* **Branch** → Result of a decision.
* **Leaf Node** → Final prediction value.
* **Split** → Division of data based on feature values.

---

## [FUN] — Functions

* **fit()** → Trains the decision tree model.
* **predict()** → Predicts output values.
* **score()** → Calculates model accuracy.

> **Clarification:** In common machine-learning libraries such as scikit-learn, `score()` for a regression model typically returns the **coefficient of determination (R²)**, rather than general "accuracy."

---

## [EXA] — Examples

* Predicting house prices based on area, location, and rooms.
* Predicting salary based on experience and skills.

---

## [USC] — Use Cases

* Real estate price prediction.
* Sales forecasting.
* Customer value prediction.
* Financial risk analysis.

---

## [CAT] — Characteristics / Considerations

* Can easily overfit with deep trees.
* Not suitable for very small datasets.
* Predictions are not smooth like linear models.

---

## [ANE] — Additional Notes / Examples

* If area < 1000 sq.ft → Price = 50 Lakhs
* If area > 1000 sq.ft → Price = 80 Lakhs

---

## Key Notes

* Decision Tree Regression splits data into smaller groups.
* Final prediction is usually the average value of samples in each leaf.
* Uses tree-based rules instead of mathematical equations.
* No need for feature normalization or scaling.

---

## Core Structure

```text id="q8m1hz"
Decision Tree Regression
        │
        ├── Root Node
        │
        ├── Decision Node
        │       └── Split based on feature values
        │
        ├── Branch
        │       └── Result of a decision
        │
        └── Leaf Node
                └── Final prediction
                    └── Average value of samples
```


# Random Forest Regression

## Definition

**Random Forest Regression** is an ensemble regression algorithm that combines multiple **Decision Trees** to make accurate predictions.

It is an **Ensemble Learning** technique that combines the predictions of multiple Decision Trees to produce a more stable and accurate prediction than a single Decision Tree.

---

## [WHY] — Why Random Forest Regression?

* Reduces overfitting compared to a single Decision Tree.
* Improves prediction accuracy by combining multiple models.
* Handles complex and non-linear relationships.

---

## [SYN] — Prediction

```text id="rfp42k"
Prediction = Average of predictions from multiple decision trees
```

For regression, the predictions from the individual Decision Trees are averaged to produce the final prediction.

---

## [PAR] — Parameters / Components

* **Tree** → Individual Decision Tree model.
* **Forest** → Collection of multiple Decision Trees.
* **n_estimators** → Number of trees in the forest.
* **Feature Selection** → Random subset of features used for each tree.

### Additional Components

* **Bootstrap Sampling** → Each Decision Tree is trained using a randomly sampled dataset, typically with replacement, from the training data.
* **Random Feature Selection** → A random subset of features is considered when creating splits in the Decision Trees.

These two sources of randomness help make the individual trees less correlated and improve the overall stability of the Random Forest.

---

## [FUN] — Functions

* **fit()** → Trains the Random Forest model.
* **predict()** → Predicts continuous output values.
* **score()** → Measures model performance.
* **feature_importances_** → Shows the importance of each feature.

> **Clarification:** In common machine-learning libraries such as scikit-learn, `score()` for a regression model typically returns the **coefficient of determination (R²)**.

---

## [EXA] — Examples

* Predicting house prices using area, location, and number of rooms.
* Predicting stock prices using multiple market indicators.

---

## [USC] — Use Cases

* Real estate price prediction.
* Financial forecasting.
* Medical prediction systems.
* Customer demand forecasting.

---

## [CAT] — Characteristics / Considerations

* Requires more computational power than a single Decision Tree.
* Less interpretable because it contains many trees.
* Large forests may increase training time.

---

## [ANE] — Additional Notes / Example

Suppose three Decision Trees produce the following predictions:

```text id="rfa17c"
Tree 1 predicts: 50 Lakhs
Tree 2 predicts: 55 Lakhs
Tree 3 predicts: 52 Lakhs
```

The final prediction is calculated by averaging the predictions:

```text id="rfa18d"
Final Prediction = (50 + 55 + 52) / 3
                 = 52.33 Lakhs
```

---

## Key Notes

* Random Forest Regression = Multiple Decision Trees + Averaging.
* Uses Bootstrap Sampling and Random Feature Selection.
* It is an Ensemble Learning technique.
* More stable and accurate than a single Decision Tree.
* No feature scaling is required.

---

## Core Structure

```text id="rfstr9"
Random Forest Regression
        │
        ├── Decision Tree 1 ──→ Prediction 1
        │
        ├── Decision Tree 2 ──→ Prediction 2
        │
        ├── Decision Tree 3 ──→ Prediction 3
        │
        ├── ...
        │
        └── Decision Tree n ──→ Prediction n
                    │
                    ▼
          Average of Predictions
                    │
                    ▼
            Final Prediction
```

### Working Concept

```text id="rfwrk2"
Training Data
     │
     ├── Bootstrap Sampling
     │       └── Tree 1
     │
     ├── Bootstrap Sampling
     │       └── Tree 2
     │
     ├── Bootstrap Sampling
     │       └── Tree 3
     │
     └── ...
             │
             ▼
    Multiple Decision Trees
             │
             ▼
    Average Predictions
             │
             ▼
      Final Prediction
```

# Support Vector Regression (SVR)

## Definition

**Support Vector Regression (SVR)** is a regression algorithm that predicts continuous values by finding the best-fit function within a **margin of tolerance**.

* SVR = Support Vector Machine applied to regression problems.
* It uses a **margin (ε-tube)** instead of minimizing every prediction error.
* Kernel functions allow SVR to model non-linear relationships.
* Feature scaling is required before training.

---

## [WHY] — Why Support Vector Regression?

* Handles non-linear relationships using kernel functions.
* Reduces the effect of outliers.
* Works well with high-dimensional datasets.
* Provides good generalization for complex data.

---

## [SYN] — Prediction Function

```text id="svr91f"
f(x) = w·x + b
```

Where:

* **w·x** represents the weighted combination of input features.
* **b** represents the bias/intercept.

> For non-linear SVR, kernel functions are used to model complex relationships. Therefore, the simple `f(x) = w·x + b` form represents the linear SVR formulation.

---

## [PAR] — Parameters

* **X** → Input features.
* **y** → Target output value.
* **w** → Weight vector.
* **b** → Bias/intercept.
* **ε (Epsilon)** → Margin of tolerance around the prediction function.
* **C** → Regularization parameter controlling the penalty for errors outside the ε-tube.
* **Kernel** → Function used to handle non-linear data.

### Important Parameter Relationship

* **ε (Epsilon)** determines the width of the tolerance margin around the prediction function.
* **C** controls how strongly errors outside the ε-tube are penalized.
* **Kernel** determines how the input data is transformed or represented to model non-linear relationships.

---

## [TYS] — Types of SVR

### Linear SVR

* Uses a straight line for prediction.
* Suitable when the relationship between features and target is approximately linear.

### Non-Linear SVR

* Uses kernel functions to fit complex patterns.
* Suitable when the relationship between features and target is non-linear.

---

## [FUN] — Functions

* **fit()** → Trains the SVR model.
* **predict()** → Predicts output values.
* **score()** → Evaluates model performance.

> **Clarification:** In common machine-learning libraries such as scikit-learn, `score()` for an SVR regression model typically returns the **coefficient of determination (R²)**.

---

## [EXA] — Examples

* Predicting stock prices.
* Predicting house prices using multiple features.
* Predicting temperature changes.

---

## [USC] — Use Cases

* Financial forecasting.
* Medical data prediction.
* Demand prediction.
* Time-series forecasting.

---

## [CAT] — Characteristics / Considerations

* Slow for very large datasets.
* Requires proper parameter tuning.
* Sensitive to feature scaling.
* Feature scaling is required before training.

---

## [ANE] — Additional Notes / Examples

### Predicting House Price

```text id="svra11"
Input  → Area, location, bedrooms
Output → House price
```

### Predicting Salary

```text id="svra12"
Input  → Experience, skills, education
Output → Expected salary
```

---

## Kernel Functions

Common kernels used with SVR:

* **Linear Kernel**
* **Polynomial Kernel**
* **RBF (Radial Basis Function) Kernel**

### Kernel Purpose

Kernel functions allow SVR to model **non-linear relationships** without explicitly creating all the higher-dimensional feature representations.

---

## Core Concept

```text id="svrc01"
Support Vector Regression (SVR)
            │
            ├── Input Features
            │
            ├── Prediction Function
            │       └── f(x) = w·x + b
            │
            ├── ε-Tube
            │       └── Margin of tolerance
            │
            ├── C
            │       └── Error penalty
            │
            └── Kernel
                    │
                    ├── Linear
                    ├── Polynomial
                    └── RBF
```

---

## Key Notes

* SVR = Support Vector Machine applied to Regression problems.
* Uses a margin (ε-tube) instead of minimizing every error.
* Kernel functions allow SVR to model non-linear relationships.
* Feature scaling is required before training.
* Common kernels:

  * Linear Kernel
  * Polynomial Kernel
  * RBF (Radial Basis Function) Kernel
