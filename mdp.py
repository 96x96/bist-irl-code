import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


FEATURES = ["Trend", "Momentum", "ROC"]
STATES = list(product([-1, 0, 1], repeat=3))
N_STATES = 27
N_ACTIONS = 3
DISCOUNT = 0.95


def load_data(path):
    data = pd.read_csv(path).rename(columns={"date": "Date"})
    columns = ["Date", *FEATURES, "Action"]
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data[columns].copy()
    data["Date"] = pd.to_datetime(data["Date"].astype(str), errors="raise")
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Use one valid, unique date per row.")
    if not data[FEATURES].isin([-1, 0, 1]).all().all():
        raise ValueError("Trend, Momentum, and ROC must be -1, 0, or 1.")
    if not data["Action"].isin([0, 1, 2]).all():
        raise ValueError("Action must be 0 (sell), 1 (hold), or 2 (buy).")
    data[FEATURES + ["Action"]] = data[FEATURES + ["Action"]].astype(int)
    state_ids = {state: i for i, state in enumerate(STATES)}
    data["State"] = [state_ids[tuple(row)] for row in data[FEATURES].to_numpy()]
    return data.sort_values("Date").reset_index(drop=True)


def split_data(data):
    split = round(0.8 * len(data))
    if split < 2 or split >= len(data):
        raise ValueError("At least two training rows and one test row are needed.")
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def transition_probabilities(data):
    probabilities = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    states = data["State"].to_numpy(dtype=int)
    actions = data["Action"].to_numpy(dtype=int)
    np.add.at(probabilities, (states[:-1], actions[:-1], states[1:]), 1)
    counts = probabilities.sum(axis=2, keepdims=True)
    probabilities = np.divide(
        probabilities, counts, out=np.zeros_like(probabilities), where=counts > 0
    )
    for state, action in np.argwhere(counts[:, :, 0] == 0):
        probabilities[state, action, state] = 1.0
    return probabilities


def value_iteration(probabilities, rewards, discount=DISCOUNT, threshold=1e-6):
    values = np.zeros(len(rewards))
    for _ in range(10000):
        q_values = rewards[:, None] + discount * (probabilities @ values)
        updated = q_values.max(axis=1)
        if np.max(np.abs(updated - values)) < threshold:
            return updated, q_values.argmax(axis=1)
        values = updated
    raise RuntimeError("Value iteration did not converge.")


def save_predictions(test, predictions, output, model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = test[["Date", "State", "Action"]].copy()
    results["Predicted_Action"] = predictions
    results.to_csv(output / f"{model}_predictions.csv", index=False)
    scores = {
        "Model": model,
        "Macro_F1": f1_score(
            test["Action"], predictions, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "Accuracy": accuracy_score(test["Action"], predictions),
    }
    pd.DataFrame([scores]).to_csv(output / f"{model}_scores.csv", index=False)
    print(f"{model}: macro F1 = {scores['Macro_F1']:.4f}, accuracy = {scores['Accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/mdp")
    args = parser.parse_args()

    train, test = split_data(load_data(args.input))
    probabilities = transition_probabilities(train)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(probabilities.reshape(N_STATES * N_ACTIONS, N_STATES))
    table.insert(0, "Action", np.tile(np.arange(N_ACTIONS), N_STATES))
    table.insert(0, "State", np.repeat(np.arange(N_STATES), N_ACTIONS))
    table.to_csv(output / "transitions.csv", index=False)
    states = pd.DataFrame(STATES, columns=FEATURES)
    states.index.name = "State"
    states.to_csv(output / "states.csv")
    print(f"Training rows: {len(train)}; test rows: {len(test)}")


if __name__ == "__main__":
    main()
