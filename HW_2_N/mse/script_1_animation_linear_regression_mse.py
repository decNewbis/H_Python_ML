import os

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from scipy.optimize import minimize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    # Initial parameters: 2, 3.5
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MSE (Mean Squared Error) Minimization Regression"""
    global mse_result_array
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat = m * x + b
    # Get loss MSE (Mean Squared Error)
    mse = (np.square(y - y_hat)).mean()
    mse_result_array.append(mse)
    return mse


def save(minimize_result: np.array) -> None:
    global minimize_result_array
    minimize_result_array.append(minimize_result)


def animate_epoch(epoch_: int) -> Tuple[np.array, ]:
    global minimize_result_array, xx, mse_
    plt.title(f'Epoch [{epoch_ + 1}/{len(minimize_result_array)}]')
    yy_ = minimize_result_array[epoch_][0] * xx + minimize_result_array[epoch_][1]
    mse_.set_data(xx, yy_)
    return mse_,


if __name__ == "__main__":
    # Initial number of samples
    nsamples_ = 100

    # Initial minimize methods
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']

    for method_ in methods:
        # Initial empty mse_result_array
        mse_result_array = []

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

        # Initial guess of the parameters: [2, 2] (m, b).
        # It doesn’t have to be accurate but simply reasonable.
        initial_guess = np.array([5, -3])

        # Maximizing the probability for point to be from the distribution
        try:
            results = minimize(
                mse_regression,
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
        xx = np.linspace(np.min(x), np.max(x), nsamples_)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.plot(x, y, "o", label="data")
        ax.plot(x, y_true, "b-", label="True")
        mse_, = ax.plot([], [], "r--.", label="MSE")
        ax.legend(loc="best")

        animation_mse = animation.FuncAnimation(
            fig,
            animate_epoch,
            frames=len(minimize_result_array),
            interval=200,
            repeat=False,
            blit=True
        )

        # Save animation
        animation_mse.save(os.path.join(BASE_DIR + f'/method_{method_}',
                                        f'mse_regression_(anim)(linear)_smpls({nsamples_}).gif'))

        # Save plot
        plt.savefig(os.path.join(BASE_DIR + f'/method_{method_}',
                                 f"mse_regression_(linear)_smpls({nsamples_}).png"))

        # Plot loss function value
        plt.figure(figsize=(8, 6))
        plt.title(f'Loss function')
        plt.xlabel('Function evaluation')
        plt.ylabel('Mean Squared Error')
        plt.plot(np.arange(0, len(mse_result_array), 1), mse_result_array)

        # Save plot
        plt.savefig(os.path.join(BASE_DIR + f'/method_{method_}',
                                 f"loss_function_mse_(linear)_smpls({nsamples_}).png"))
        plt.close()
