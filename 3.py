# Practical 3: Bayesian Network for Heart Disease Diagnosis (pgmpy)

import pandas as pd

try:
    from pgmpy.models import DiscreteBayesianNetwork                        # newer pgmpy
except ImportError:
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork     # older pgmpy
from pgmpy.inference import VariableElimination

# Step 1: Load the dataset
df = pd.read_csv("data/heart_disease_uci.csv")

# Step 2: Clean and encode the data
df = df.rename(columns={"thalch": "thalach", "num": "target", "HeartDisease": "target"})
df["sex"] = df.sex.replace({"Male": 1, "Female": 0})
df["target"] = (df.target > 0).astype(int)                 # any disease level -> 1
df["cp"] = df.cp.astype("category").cat.codes              # chest pain text -> numbers
df["exang"] = df.exang.astype(str).str.lower().map(
    {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})

df = df[["age", "sex", "cp", "trestbps", "chol", "exang", "thalach", "target"]].dropna()
df = df[df.chol > 0].astype(float)                         # chol = 0 means "not recorded"

# Step 3: Discretize continuous columns into 3 bins (0, 1, 2)
bins = {
    "age":      [45, 60],
    "chol":     [200, 240],
    "trestbps": [120, 140],
    "thalach":  [120, 150],
}
for col, cuts in bins.items():
    edges = [-float("inf")] + cuts + [float("inf")]
    df[col] = pd.cut(df[col], edges, labels=[0, 1, 2]).astype(int)
df = df.astype(int)

# Step 4: Define the structure (risk factors -> heart disease -> symptoms)
model = DiscreteBayesianNetwork([
    ("age", "target"), ("sex", "target"), ("chol", "target"), ("trestbps", "target"),
    ("target", "cp"), ("target", "exang"), ("target", "thalach"),
])

# Step 5: Learn the probability tables (maximum likelihood)
model.fit(df)
infer = VariableElimination(model)

# Step 6: Diagnose patients
cases = {
    "older male, chest pain=0, exercise angina": {"age": 2, "sex": 1, "cp": 0, "exang": 1, "thalach": 0},
    "young female, no symptoms":                {"age": 0, "sex": 0, "cp": 2, "exang": 0, "thalach": 2},
}
for name, evidence in cases.items():
    p = infer.query(["target"], evidence=evidence, show_progress=False)
    print(f"{name}: P(heart disease) = {p.values[1]:.2f}")