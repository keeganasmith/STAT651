import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.stats import t

import math
#type 1 error: reject the null when it was actually true
#type 2 error: fail to reject the null, when null was false

def split_by(df, col):
    unique_col_values = df[col].unique()
    df_dict = {}
    for value in unique_col_values:
        df_dict[value] = df.loc[df[col] == value]
    return df_dict

def standard_deviation(df, col):
    return df.std(ddof = 1)[col]

def median(df, col):
    return df.median()[col]

def find_outliers(df, col, factor = 1.5):
    first_quantile = df[col].quantile(.25)
    third_quantile = df[col].quantile(.75)
    IQR = third_quantile - first_quantile
    lower_bound = first_quantile - factor * IQR
    upper_bound = third_quantile + factor * IQR
    outliers = df.loc[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

def get_percentile_inclusive(df, col, value):
    sorted_values = df[col].sort_values()
    count_leq = (sorted_values <= value).sum()
    percentile = count_leq / len(sorted_values)
    return percentile

def get_percentile_exclusive(df, col, value):
    sorted_values = df[col].sort_values()
    count_le = (sorted_values < value).sum()
    percentile = count_le / len(sorted_values)
    return percentile

def get_iqr(df, col):
    first_quantile = df[col].quantile(.25)
    third_quantile = df[col].quantile(.75)
    IQR = third_quantile - first_quantile
    return IQR

def histogram(df, col):
    x = df[col]
    counts, bins = np.histogram(x)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

def compute_z_score(value, mean, std):
    return (value - mean) / std

def confidence_interval_z(mean, n, std, confidence=0.95):
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_critical * (std / np.sqrt(n))

    lower = mean - margin_of_error
    upper = mean + margin_of_error
    return lower, upper

def t_p_value(null_hypoth, sample_mean, n, df, std):
    t_stat = (sample_mean - null_hypoth) / (std / math.sqrt(n))
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return p_value

def t_test(null_hypoth, sample_mean, n, std, alpha):
    df = n - 1
    t_critical = solve_for_t_half(alpha, df)
    t_stat = (sample_mean - null_hypoth) / (std / math.sqrt(n))
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    reject = p_value < alpha
    return reject, t_critical, t_stat, p_value

def confidence_interval_t(mean, n, std, confidence=0.95):
    df = n - 1
    t_critical = t.ppf((1 + confidence) / 2, df)
    margin_of_error = t_critical * (std / math.sqrt(n))
    lower = mean - margin_of_error
    upper = mean + margin_of_error
    return lower, upper
#P(x <= z_score)
def cdf(z_score):
    cdf = norm.cdf(z_score)
    return cdf

def inverse_cdf(probability):
    return norm.ppf(probability)

def compute_standard_error(sigma, n):
    return sigma / math.sqrt(n)
#zα/2 is defined as the number for N(0, 1) such that
#P(Z > zα/2) = α/2. This value is used for two-sided testing.
def solve_for_z_half(alpha):
    probability = 1 - alpha / 2
    z_half = inverse_cdf(probability)
    return z_half

def solve_for_t_half(alpha, degree_f):
    t_alpha_over_2 = t.ppf(1 - alpha/2, degree_f)
    return t_alpha_over_2

def compute_required_sample_size(alpha, sigma, error):
    z_a_half = solve_for_z_half(alpha)
    return (z_a_half * sigma / error)**2

def compute_sample_size_power(alpha, beta, delta, sigma):
    #power = 1 - beta
    z_alpha_half = solve_for_z_half(alpha)
    z_beta = abs(inverse_cdf(1 - beta))
    required_sample_size = sigma**2 / delta**2 * (z_alpha_half + z_beta)**2
    return required_sample_size

def compute_pooled_std(n1, n2, s1, s2):
    return math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))