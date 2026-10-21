# BIST Trading Behavior

Code for *[Inferring Latent Trading Motivations in Borsa Istanbul: A Comparative Inverse Reinforcement Learning Approach](https://doi.org/10.1016/j.bir.2026.100834)*.

| File | Purpose |
| --- | --- |
| `mdp.py` | 27 states and empirical transition probabilities |
| `lirl.py` | Linear inverse reinforcement learning |
| `maxent_irl.py` | Maximum entropy inverse reinforcement learning |
| `bc.py` | Logistic regression, random forest, and KNN |
| `example.py` | Generate synthetic data for a quick run |

Python 3.11 or newer:

```bash
pip install -r requirements.txt
python example.py
python mdp.py --input data/example.csv
python lirl.py --input data/example.csv
python maxent_irl.py --input data/example.csv
python bc.py --input data/example.csv
```

Use one CSV per stock and broker. See [data/README.md](data/README.md) for the format.
Replace `data/example.csv` with your local input path and use `--output` to choose a result folder.

Rows are sorted by date. The first 80% are used for training and the last 20% for testing.
Transitions, expert policies, trajectories, and parameter selection use training data only.
IRL uses a discount factor of 0.95 and one reward per state. Unobserved state-action pairs use a self-loop.
LIRL selects its L1 penalty over 0.00 to 4.99 using agreement with the training expert policy.
MaxEnt uses 1,250 trajectories of length 20, 1,000 updates, and a learning rate of 0.01.
BC uses five chronological cross-validation folds within the training set.
Scores are macro F1 over sell, hold, and buy, and accuracy. Results are saved under `outputs/`.

The study data cannot be redistributed and are not included. The example is entirely synthetic.
Data files and generated results are excluded from Git.

Methods: [Ng and Russell (2000)](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf), [Ziebart et al. (2008)](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html).

## License

The code in this repository is licensed under the [MIT License](LICENSE).
This software license does not cover the accompanying paper or the original study data.
