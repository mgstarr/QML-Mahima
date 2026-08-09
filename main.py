!pip -q install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression

# Permutation Importance
from sklearn.inspection import permutation_importance




breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Features and target
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets




print("========== DATASET METADATA ==========")
print(breast_cancer_wisconsin_diagnostic.metadata)

print("\n========== VARIABLE INFORMATION ==========")
print(breast_cancer_wisconsin_diagnostic.variables)


print("\n========== DATASET SHAPE ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\n========== FEATURE NAMES ==========")
print(X.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
display(X.head())

print("\n========== TARGET VALUES ==========")
display(y.head())




target_column = y.columns[0]

df = pd.concat([X, y], axis=1)

print("\n========== COMPLETE DATASET ==========")
display(df.head())

print("\nDataset shape:", df.shape)


# ============================================================
# 6. DATA TYPES AND BASIC INFORMATION
# ============================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFO ==========")
df.info()



print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", df.isnull().sum().sum())



print("\n========== DUPLICATE ROWS ==========")

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)




print("\n========== DESCRIPTIVE STATISTICS ==========")

display(X.describe().T)



print("\n========== TARGET DISTRIBUTION ==========")

print(y[target_column].value_counts())

print("\nTarget percentages:")

print(
    y[target_column]
    .value_counts(normalize=True) * 100
)



plt.figure(figsize=(7, 5))

sns.countplot(
    x=target_column,
    data=df
)

plt.title("Distribution of Breast Cancer Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")

plt.show()


X.hist(
    figsize=(18, 20),
    bins=30
)

plt.suptitle(
    "Distribution of Numerical Features",
    fontsize=16
)

plt.tight_layout()
plt.show()



plt.figure(figsize=(18, 8))

sns.boxplot(data=X)

plt.xticks(rotation=90)

plt.title("Box Plot of Features")
plt.xlabel("Features")
plt.ylabel("Values")

plt.show()



correlation_matrix = X.corr()

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.show()



upper = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

high_correlations = upper.stack().reset_index()

high_correlations.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

high_correlations = high_correlations[
    high_correlations["Correlation"].abs() >= 0.80
]

high_correlations = high_correlations.sort_values(
    by="Correlation",
    key=abs,
    ascending=False
)

print("\n========== HIGHLY CORRELATED FEATURES ==========")

display(high_correlations)



selected_features = [
    col for col in [
        "radius1",
        "radius2",
        "radius3",
        "texture1",
        "perimeter1",
        "area1",
        "smoothness1",
        "concavity1"
    ]
    if col in X.columns
]

print("\nSelected features:")
print(selected_features)

# If exact column names differ, use first 6 features
if len(selected_features) == 0:
    selected_features = X.columns[:6].tolist()


for feature in selected_features:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        x=target_column,
        y=feature,
        data=df
    )

    plt.title(f"{feature} by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel(feature)

    plt.show()


pairplot_features = selected_features[:4] + [target_column]

sns.pairplot(
    df[pairplot_features],
    hue=target_column
)

plt.show()


print("\nOriginal target values:")
print(y[target_column].unique())

# UCI dataset:
# M = Malignant
# B = Benign

if y[target_column].dtype == "object":

    y_encoded = y[target_column].map({
        "M": 1,
        "B": 0
    })

else:

    y_encoded = y[target_column]


print("\nEncoded target distribution:")
print(y_encoded.value_counts())



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)




y_pred = model.predict(X_test_scaled)




accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")



print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)



cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
    "Absolute Coefficient": np.abs(model.coef_[0])
})

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\n========== LOGISTIC REGRESSION FEATURE IMPORTANCE ==========")

display(coefficients.head(15))



#permutation importance


print("\n========== PERMUTATION IMPORTANCE ANALYSIS ==========")


perm_importance = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="accuracy"
)




permutation_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance Mean": perm_importance.importances_mean,
    "Importance Std": perm_importance.importances_std
})

# Sort from most important to least important
permutation_df = permutation_df.sort_values(
    by="Importance Mean",
    ascending=False
)

print("\n========== RANKED PERMUTATION IMPORTANCE ==========")

display(permutation_df)




top_permutation = permutation_df.head(15)

print("\n========== TOP 15 FEATURES ==========")

display(top_permutation)



plt.figure(figsize=(10, 8))

plt.barh(
    top_permutation["Feature"][::-1],
    top_permutation["Importance Mean"][::-1],
    xerr=top_permutation["Importance Std"][::-1]
)

plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")

plt.title(
    "Top 15 Features - Permutation Importance"
)

plt.tight_layout()

plt.show()



plt.figure(figsize=(10, 12))

plt.barh(
    permutation_df["Feature"][::-1],
    permutation_df["Importance Mean"][::-1],
    xerr=permutation_df["Importance Std"][::-1]
)

plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")

plt.title(
    "Permutation Importance of All Features"
)

plt.tight_layout()

plt.show()



comparison_df = coefficients[
    ["Feature", "Coefficient", "Absolute Coefficient"]
].merge(
    permutation_df[
        ["Feature", "Importance Mean", "Importance Std"]
    ],
    on="Feature"
)

comparison_df = comparison_df.sort_values(
    by="Importance Mean",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE COMPARISON ==========")

display(comparison_df)


print("\n==============================================")
print("              ANALYSIS SUMMARY")
print("==============================================")

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(
    f"Missing values: "
    f"{df.isnull().sum().sum()}"
)
print(
    f"Duplicate rows: "
    f"{df.duplicated().sum()}"
)

print("\nTarget distribution:")

print(
    y[target_column].value_counts()
)

print(
    f"\nLogistic Regression Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nTop 10 features according to permutation importance:")

display(
    permutation_df.head(10)
)

print("\nAnalysis completed successfully!")
