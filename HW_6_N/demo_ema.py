from typing import Text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.DataFrame(
        {"data": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    ema = data.ewm(com=0.99).mean()
    mma = data.rolling(3).mean()

    # Comparison plot b/w stock values & EMA
    plt.plot(data["data"], "ob", label="data")
    plt.plot(ema, label="EMA Values", color="green")
    plt.plot(mma, label="MMA Values", color="magenta")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(f"simple_data.png")
    plt.figure()

    ##################################################

    x = np.linspace(start=-5, stop=3, num=80)
    y = (x**3 + 3*x**2 - 6*x - 8) / 4
    y = y + np.random.uniform(-2, 2, size=x.shape)
    
    data = pd.DataFrame(
        {"data": y}
    )
    ema = data.ewm(alpha=0.05).mean()
    mma = data.rolling(3).mean()

    # Comparison plot b/w stock values & EMA
    plt.plot(data["data"], "ob", label="data")
    plt.plot(ema, label="EMA Values", color="green")
    plt.plot(mma, label="MMA Values", color="magenta")
    plt.xlabel("iteration")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(f"noisy_data.png")
    plt.figure()
