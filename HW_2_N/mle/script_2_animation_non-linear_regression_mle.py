import os

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy

from matplotlib import animation
from scipy.optimize import minimize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    # Initial parameters: 2, 3.5
    y = 2 * x**2 + x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mle_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """Maximum Likelihood Estimation Regression"""
    global mle_result_array
    m = guess[0]
    b = guess[1]
    sigma = guess[2]
    # Predictions
    y_hat = m * x**2 + x + b
    # Compute PDF (probability density function) of observed values normally distributed around mean (y_hat)
    # with a standard deviation of sigma
    # Must watch: https://www.youtube.com/watch?v=Dn6b9fCIUpM
    # logpdf: logarithm probability density function
    neg_ll = -np.sum(scipy.stats.norm.logpdf(y, loc=y_hat,
                     scale=sigma))  # return negative LL (log likelihood)
    mle_result_array.append(neg_ll)
    return neg_ll


def save(minimize_result: np.array) -> None:
    global minimize_result_array
    minimize_result_array.append(minimize_result)


def animate_epoch(epoch_: int) -> Tuple[np.array, ]:
    global minimize_result_array, xx, mle_
    plt.title(f'Epoch [{epoch_ + 1}/{len(minimize_result_array)}]')
    yy_ = minimize_result_array[epoch_][0] * xx**2 + xx + minimize_result_array[epoch_][1]
    mle_.set_data(xx, yy_)
    return mle_,


if __name__ == "__main__":
    # Initial number of samples
    nsamples_ = 100

    # Initial minimize methods
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']

    for method_ in methods:
        # Initial empty mle_result_array
        mle_result_array = []

        # Initial empty minimize_result_array
        minimize_result_array = []

        # Getting data
        x, y_true = get_data(nsamples_)
        y = add_noise(y_true)

        # Plot and investigate data
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.plot(x, y, "o", label="data")
        ax.legend(loc="best")
        plt.savefig("data.png")

        # Initial guess of the parameters: [2, 2, 2] (m, b, sigma).
        # It doesn’t have to be accurate but simply reasonable.
        initial_guess = np.array([2, 2, 2])

        # Maximizing the probability for point to be from the distribution
        try:
            results = minimize(
                mle_regression,
                initial_guess,
                args=(x, y,),
                method=method_,
                options={"disp": True},
                callback=save
            )
        except ValueError:
            continue
        except TypeError:
            continue
        print(results)
        print(f"Method [{method_}]. Parameters: ", results.x)

        # Make dir for current method if it not exists
        if not os.path.exists(BASE_DIR + f'method_{method_}'):
            os.makedirs(BASE_DIR + f'/method_{method_}', exist_ok=True)

        # Plot and animate results
        xx = np.linspace(np.min(x), np.max(x), 100)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.plot(x, y, "o", label="data")
        ax.plot(x, y_true, "b-", label="True")
        mle_, = ax.plot([], [], "r--.", label="MLE")
        ax.legend(loc="best")

        animation_mle = animation.FuncAnimation(
            fig,
            animate_epoch,
            frames=len(minimize_result_array),
            interval=200,
            repeat=False,
            blit=True
        )
        # Save animation
        animation_mle.save(os.path.join(BASE_DIR + f'/method_{method_}',
                                        f'mle_regression_(anim)(non-linear)_smpls({nsamples_}).gif'))

        # Save plot
        plt.savefig(os.path.join(BASE_DIR + f'/method_{method_}',
                                 f"mle_regression_(non-linear)_smpls({nsamples_}).png"))

        # Plot loss function value
        plt.figure(figsize=(8, 6))
        plt.title(f'Loss function')
        plt.xlabel('Function evaluation')
        plt.ylabel('Maximum Likelihood Estimation')
        plt.plot(np.arange(0, len(mle_result_array), 1), mle_result_array)

        # Save plot
        plt.savefig(os.path.join(BASE_DIR + f'/method_{method_}',
                                 f"loss_function_mle_(non-linear)_smpls({nsamples_}).png"))
        plt.close()
