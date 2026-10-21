import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from mdp import (
    DISCOUNT, N_ACTIONS, N_STATES, load_data, save_predictions,
    split_data, transition_probabilities, value_iteration,
)


def expert_policy(data):
    counts = pd.crosstab(data["State"], data["Action"]).reindex(
        index=range(N_STATES), columns=range(N_ACTIONS), fill_value=0
    )
    return counts.to_numpy().argmax(axis=1)


def linear_irl(probabilities, policy, l1):
    n_states, n_actions, _ = probabilities.shape
    identity = np.eye(n_states)
    expert_transitions = probabilities[np.arange(n_states), policy]
    future = np.linalg.solve(identity - DISCOUNT * expert_transitions, identity)
    constraints = []
    for state in range(n_states):
        for action in range(n_actions):
            if action != policy[state]:
                gap = (expert_transitions[state] - probabilities[state, action]) @ future
                constraints.append(np.r_[-gap, identity[state], np.zeros(n_states)])
    constraints.extend(np.hstack([identity, np.zeros_like(identity), -identity]))
    constraints.extend(np.hstack([-identity, np.zeros_like(identity), -identity]))
    objective = np.r_[np.zeros(n_states), -np.ones(n_states), l1 * np.ones(n_states)]
    bounds = [(-1, 1)] * n_states + [(0, None)] * (2 * n_states)
    result = linprog(
        objective,
        A_ub=np.asarray(constraints),
        b_ub=np.zeros(len(constraints)),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:n_states]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/lirl")
    parser.add_argument("--l1", type=float)
    args = parser.parse_args()
    if args.l1 is not None and args.l1 < 0:
        parser.error("--l1 must be nonnegative.")

    train, test = split_data(load_data(args.input))
    probabilities = transition_probabilities(train)
    policy = expert_policy(train)
    penalties = np.arange(0, 5, 0.01) if args.l1 is None else [args.l1]
    best_score = -1.0
    for penalty in penalties:
        rewards = linear_irl(probabilities, policy, penalty)
        _, recovered_policy = value_iteration(probabilities, rewards)
        score = np.mean(recovered_policy == policy)
        if score > best_score:
            best_score = score
            best_rewards = rewards
            best_policy = recovered_policy
            best_penalty = penalty

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "State": np.arange(N_STATES), "Reward": best_rewards, "Policy": best_policy,
    }).to_csv(output / "lirl_rewards.csv", index=False)
    print(f"L1: {best_penalty:.2f}")
    save_predictions(test, best_policy[test["State"].to_numpy()], output, "lirl")


if __name__ == "__main__":
    main()
