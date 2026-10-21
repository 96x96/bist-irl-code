import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from mdp import FEATURES, load_data, save_predictions, split_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/bc")
    args = parser.parse_args()

    train, test = split_data(load_data(args.input))
    if len(train) < 60 or train["Action"].nunique() < 2:
        raise ValueError("BC needs at least 60 training rows and two action classes.")
    models = {
        "LR": (
            OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=100)),
            {"estimator__C": [0.1, 1.0, 10.0]},
        ),
        "RF": (
            RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
            {"criterion": ["entropy", "gini"]},
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
            {"n_neighbors": [3, 5, 7]},
        ),
    }
    scorer = make_scorer(f1_score, labels=[0, 1, 2], average="macro", zero_division=0)
    for name, (model, parameters) in models.items():
        search = GridSearchCV(
            model, parameters, scoring=scorer, cv=TimeSeriesSplit(n_splits=5),
            error_score="raise",
        )
        search.fit(train[FEATURES], train["Action"])
        predictions = search.predict(test[FEATURES])
        print(f"{name}: {search.best_params_}")
        save_predictions(test, predictions, args.output, name)


if __name__ == "__main__":
    main()
