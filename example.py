from pathlib import Path

import numpy as np
import pandas as pd


def main():
    random = np.random.RandomState(42)
    features = random.choice([-1, 0, 1], size=(600, 3))
    actions = np.sign(features[:, 0] + features[:, 1]).astype(int) + 1
    noisy = random.rand(len(actions)) < 0.2
    actions[noisy] = random.randint(0, 3, size=noisy.sum())
    data = pd.DataFrame(features, columns=["Trend", "Momentum", "ROC"])
    data.insert(0, "Date", pd.bdate_range("2020-01-01", periods=len(data)))
    data["Action"] = actions
    Path("data").mkdir(exist_ok=True)
    data.to_csv("data/example.csv", index=False)
    print("Created data/example.csv (synthetic data).")


if __name__ == "__main__":
    main()
