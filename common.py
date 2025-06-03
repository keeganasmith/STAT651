import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats

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

def find_outliers(df, col):
    first_quantile = df[col].quantile(.25)
    third_quantile = df[col].quantile(.75)
    IQR = third_quantile - first_quantile
    lower_bound = first_quantile - 1.5 * IQR
    upper_bound = third_quantile + 1.5 * IQR
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
#P(x <= z_score)
def cdf(z_score):
    cdf = norm.cdf(z_score)
    return cdf