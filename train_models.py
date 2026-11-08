"""Train, tune and evaluate the five classifiers; save models, metrics and test data."""
import json
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from features import build_features, TARGET
from evaluation import evaluate  # same code path as the Streamlit app (label = P(TV) >= 0.5)

SEED = 2026
raw = pd.read_csv("data/netflix_titles.csv")
data = build_features(raw)
print("feature matrix:", data.shape, "| TV-show share: %.3f" % data[TARGET].mean())

X, y = data.drop(columns=[TARGET]), data[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)
print("train", X_tr.shape, "test", X_te.shape)

def pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])

candidates = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
                      {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 20, 50]}),
    "kNN": (pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}),
    "Naive Bayes": (pipe(GaussianNB()), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
    "Support Vector Machine": (pipe(SVC(kernel="rbf", probability=True, class_weight="balanced",
                                        random_state=SEED)),
                               {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]}),
}
FILES = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
         "kNN": "knn", "Naive Bayes": "naive_bayes", "Support Vector Machine": "svm"}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, best_params = [], {}
for name, (est, grid) in candidates.items():
    gs = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    model = gs.best_estimator_
    metrics, _, _ = evaluate(model, X_te, y_te)
    rows.append({"ML Model Name": name, **metrics})
    best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    joblib.dump(model, f"model/{FILES[name]}.joblib")
    print(name, best_params[name], {k: round(v, 4) for k, v in rows[-1].items() if k != "ML Model Name"})

res = pd.DataFrame(rows)
res.round(4).to_csv("model/metrics_summary.csv", index=False)
json.dump({"best_params": best_params, "seed": SEED, "n_train": len(X_tr), "n_test": len(X_te)},
          open("model/training_info.json", "w"), indent=2, default=str)
print(res.round(4).to_string(index=False))
