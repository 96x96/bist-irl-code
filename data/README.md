# Input format

Provide one CSV per stock and broker, with one row per trading day.

| Column | Values |
| --- | --- |
| `Date` | `YYYY-MM-DD` (`date` is also accepted) |
| `Trend` | -1, 0, 1 |
| `Momentum` | -1, 0, 1 |
| `ROC` | -1, 0, 1 |
| `Action` | 0 = sell, 1 = hold, 2 = buy |

Trend is the sign of the sum of SMA, MACD, and Bollinger Band signals.
Momentum is the sign of the sum of RSI, Stochastic, and Williams %R signals.
ROC is -1 at or below -1.5%, 1 at or above 1.5%, and 0 otherwise.
States enumerate `(Trend, Momentum, ROC)` from `(-1, -1, -1)` to `(1, 1, 1)` as 0 to 26.

Keep private inputs here or provide an external path with `--input`. CSV files are ignored by Git.
