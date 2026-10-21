import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mdp import (
    DISCOUNT, N_STATES, load_data, save_predictions,
    split_data, transition_probabilities, value_iteration,
)


def sample_trajectories(data, count, length):
    if length > len(data):
        raise ValueError("Trajectory length exceeds the training data length.")
    random = np.random.RandomState(42)
    starts = random.randint(0, len(data) - length + 1, size=count)
    states = data["State"].to_numpy(dtype=int)
    return np.array([states[start:start + length] for start in starts])


def maxent_irl(probabilities, trajectories, epochs=1000, learning_rate=0.01):
    n_states = probabilities.shape[0]
    random = np.random.RandomState(42)
    rewards = random.uniform(size=n_states)
    observed = np.bincount(trajectories.ravel(), minlength=n_states) / len(trajectories)
    start = np.bincount(trajectories[:, 0], minlength=n_states) / len(trajectories)

    for _ in range(epochs):
        values, _ = value_iteration(probabilities, rewards, threshold=1e-2)
        q_values = rewards[:, None] + DISCOUNT * (probabilities @ values)
        policy = np.exp(q_values - q_values.max(axis=1, keepdims=True))
        policy /= policy.sum(axis=1, keepdims=True)
        policy_transitions = np.einsum("sa,sak->sk", policy, probabilities)
        expected = np.zeros(n_states)
        distribution = start.copy()
        for _ in range(trajectories.shape[1]):
            expected += distribution
            distribution = distribution @ policy_transitions
        rewards += learning_rate * (observed - expected)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/maxent")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--trajectories", type=int, default=1250)
    parser.add_argument("--length", type=int, default=20)
    args = parser.parse_args()
    if min(args.epochs, args.trajectories, args.length) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, trajectories, length, and learning rate must be positive.")

    train, test = split_data(load_data(args.input))
    probabilities = transition_probabilities(train)
    trajectories = sample_trajectories(train, args.trajectories, args.length)
    rewards = maxent_irl(probabilities, trajectories, args.epochs, args.learning_rate)
    _, policy = value_iteration(probabilities, rewards)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "State": np.arange(N_STATES), "Reward": rewards, "Policy": policy,
    }).to_csv(output / "maxent_rewards.csv", index=False)
    save_predictions(test, policy[test["State"].to_numpy()], output, "maxent")


if __name__ == "__main__":
    main()
