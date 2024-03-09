import numpy as np
import pandas as pd
import scipy.stats as sts
from tqdm import tqdm


"""In this script we generate the data used to train the neural networks for pricing and implied volatility
We consider the interest free rate to be equal to 0"""


# ************************************************ Implied Volatility ************************************************


# Price generator
def black_scholes(moneyness: float, ttm: float, vol: float) -> float:
    """Black Scholes formula for European Call option price normalized by strike
    Args:
        - moneyness: spot over strike
        - ttm: time to maturity
        - vol: volatility
    Returns:
        - float: price of the Call option"""
    if vol == 0:
        return np.maximum(moneyness - 1, 0)
    d1 = np.log(moneyness) / (np.sqrt(ttm) * vol) + 0.5 * np.sqrt(ttm) * vol
    d2 = d1 - np.sqrt(ttm) * vol

    return moneyness * sts.norm.cdf(d1) - sts.norm.cdf(d2)


# Sampling the datapoints
n_samples = 2000000
sampling_space = dict(moneyness=(0.5, 1.4), ttm=(0.2, 10), volatility=(0.1, 1.0))
sampler = sts.qmc.Halton(d=3, scramble=True, seed=42)
sample = sampler.random(n=n_samples)

sample = np.array(
    [
        sampling_space["moneyness"][1] - sampling_space["moneyness"][0],
        sampling_space["ttm"][1] - sampling_space["ttm"][0],
        sampling_space["volatility"][1] - sampling_space["volatility"][0],
    ]
) * sample + np.array(
    [
        sampling_space["moneyness"][0],
        sampling_space["ttm"][0],
        sampling_space["volatility"][0],
    ]
)

print(np.min(sample, axis=0), np.max(sample, axis=0))


normalized_price = np.zeros(n_samples)
# Final range for log normalized time value is around (-16.1177118482979, -0.3932281552817425)
for i in tqdm(range(sample.shape[0])):
    normalized_price[i] = black_scholes(*sample[i])

time_value = normalized_price - np.maximum(sample[:, 0] - 1, 0)
indices = time_value > 1e-7

X = np.concatenate(
    (
        sample[indices, 0].reshape(-1, 1),
        sample[indices, 1].reshape(-1, 1),
        np.log(time_value[indices]).reshape(-1, 1),
    ),
    axis=1,
)
Y = sample[indices, 2].reshape(-1, 1)

np.save("X", X)
np.save("Y", Y)


# *******************************************************************************************************************
