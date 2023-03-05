import json

import numpy as np
import matplotlib.pyplot as plt


SETTINGS_FILE = 'experiment_settings.json'
G_ = 9.81  # Acceleration of gravity


def get_settings(setting_file: str):
    with open(setting_file, encoding='utf-8') as content_db:
        result = json.load(content_db)
    return result


def generate_normal_data(value_name: str, distribution_: dict):
    loc_ = distribution_.get(value_name)
    scale_ = distribution_.get('scale')
    size_ = distribution_.get('size')
    return np.random.normal(loc_, scale_, size_)


def generate_uniform_data(low_value_name: str, high_value_name: str, distribution_: dict):
    low_ = distribution_.get(low_value_name)
    high_ = distribution_.get(high_value_name)
    size_ = distribution_.get('size')
    return np.random.uniform(low_, high_, size_)


def get_flight_range(v0, angle):
    result = (v0 ** 2 * np.sin(2 * np.radians(angle))) / G_
    return result


def fig_conf(v0_d_name_: str, alpha_d_name: str):
    plt.xlabel('Flight length (L)')
    plt.ylabel('Frequency (n)')
    plt.title(f'Distribution(V0 {v0_d_name_}, Alpha {alpha_d_name})')


if __name__ == '__main__':
    ax = []
    normal_distribution = get_settings(SETTINGS_FILE).get('normal distribution')
    uniform_distribution = get_settings(SETTINGS_FILE).get('uniform distribution')

    v0_normal_data = generate_normal_data('v0_value', normal_distribution)
    alpha_normal_data = generate_normal_data('alpha_value', normal_distribution)

    v0_uniform_data = generate_uniform_data('v0_low', 'v0_high', uniform_distribution)
    alpha_uniform_data = generate_uniform_data('alpha_low', 'alpha_high', uniform_distribution)

    flight_range_normal_normal = get_flight_range(v0_normal_data, alpha_normal_data)
    flight_range_uniform_uniform = get_flight_range(v0_uniform_data, alpha_uniform_data)
    flight_range_normal_uniform = get_flight_range(v0_normal_data, alpha_uniform_data)
    flight_range_uniform_normal = get_flight_range(v0_uniform_data, alpha_normal_data)

    plt.figure(figsize=(15, 10))
    ax.append(plt.subplot(2, 2, 1))
    plt.hist(flight_range_normal_normal)
    fig_conf('normal', 'normal')

    ax.append(plt.subplot(2, 2, 2))
    plt.hist(flight_range_uniform_uniform)
    fig_conf('uniform', 'uniform')

    ax.append(plt.subplot(2, 2, 3))
    plt.hist(flight_range_normal_uniform)
    fig_conf('normal', 'uniform')

    ax.append(plt.subplot(2, 2, 4))
    plt.hist(flight_range_uniform_normal)
    fig_conf('uniform', 'normal')

    plt.tight_layout(h_pad=2, w_pad=4)
    for element in ax:
        element.grid()
    plt.show()
