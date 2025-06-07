import pandas as pd
import sys
import os
import math
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from common import standard_deviation, split_by, median, find_outliers, get_percentile_exclusive, get_iqr, histogram, compute_z_score, cdf, confidence_interval_z, compute_required_sample_size, compute_standard_error, confidence_interval_t, t_test, t_p_value
def problem_set_1():
    mean = 2.8
    std = 2
    sample = 4.8
    z_score = compute_z_score(sample, mean, std)
    print("probability > 4.8: ", 1 - cdf(z_score))
    print("probability < 4.8: ", cdf(z_score))
    sample = 2.8
    z_score = compute_z_score(sample, mean, std)
    print("probability > 2.8: ", 1 - cdf(z_score))
    print("probability < 2.8: ", cdf(z_score))
    print("probability 2.8 < X < 4.8: ", cdf(compute_z_score(4.8, mean, std)) - .5)

def problem_set_2():
    mean = 4.8
    std = 2
    print("probability < 2.8: ", cdf(compute_z_score(2.8, mean, std)))

def problem_set_3():
    blue_df = pd.read_csv("./hw1/bluebonnet.csv")
    close_df = blue_df.loc[blue_df["distance"] == 0]
    sample_mean = close_df["redpetals"].mean()
    print("mean is: ", sample_mean)
    z_score = compute_z_score(sample_mean, 3, 1.5)
    print("z stat: ", z_score)
    lower, upper = confidence_interval_z(sample_mean, len(close_df), 1.5, .95)
    print("confidence interval: ", lower, ",", upper)
    lower, upper = confidence_interval_z(sample_mean, len(close_df), 1.5, .99)
    print("confidence interval: ", lower, ",", upper)

def problem_set_4():
    blue_df = pd.read_csv("./hw1/bluebonnet.csv")
    far_df = blue_df.loc[blue_df["distance"] == 1]
    sigma = 1.5
    alpha = .05
    error = .2
    sample_size = compute_required_sample_size(alpha, sigma, error)
    print("required sample size for .95 with error .2: ", sample_size)

def problem_set_5():
    #The breaking strengths for a 1-square-foot sample of a particular synthetic fabric are approximately Normally distributed with a mean of 2250 pounds per square inch (PSI) and standard deviation of 10.2 PSI. 
    mean = 2250
    std = 10.2
    z_score = compute_z_score(2240, mean, std)
    p_greater_than_2240 = 1 - cdf(z_score)
    print("prob greater than 2240: ", p_greater_than_2240)
    n = 18
    SE = compute_standard_error(std, n)
    print("standard error: ", SE)
    
    start, end = confidence_interval_z(2240, n, std)
    print("95% confidence: ", start, ", ", end)

def problem_set_6():
    mu = 2.35
    sigma = 1.1
    raw_chol = 250
    X = math.log10(250)
    z_score = compute_z_score(X, mu, sigma)
    prob_greater = 1 - cdf(z_score)
    print("prob greater than 250: ", prob_greater)
    
    raw_lower = 150
    raw_upper = 250
    lower_x = math.log10(raw_lower)
    upper_x = math.log10(raw_upper)
    lower_z = compute_z_score(lower_x, mu, sigma)
    upper_z = compute_z_score(upper_x, mu, sigma)
    prob = cdf(upper_z) - cdf(lower_z)
    print("probability between 150 and 250: ", prob)

    raw_chol = 300
    X = math.log10(raw_chol)
    z_score = compute_z_score(X, mu, sigma)
    prob_greater = 1 - cdf(z_score)
    print("probability greater than 300: ", prob_greater)

def problem_set_7():
    adduct_df = pd.read_csv("./hw1/adduct.csv")
    adduct_df["distal_log"] = np.log(adduct_df["distal"])
    print(adduct_df["distal_log"])
    # histogram(adduct_df, "distal")
    # histogram(adduct_df, "distal_log")
    std = adduct_df["distal"].std()
    print("sample std: ", std)
    log_std = adduct_df["distal_log"].std()
    print("log std: ", log_std)

    n = len(adduct_df)
    SE = compute_standard_error(std, n)
    print("standard error: ", SE)

    log_SE = compute_standard_error(log_std, n)
    print("log standard error: ", log_SE)    

def problem_set_8():
    hormone_df = pd.read_csv("./hw1/hormone_assay.csv")
    #column of interest: Log(Test / Reference)
    column = "Log(Test / Reference)"
    std = hormone_df[column].std()
    mean = hormone_df[column].mean()
    print("mean is: ", mean)
    n = len(hormone_df)
    start, end = confidence_interval_t(mean, n, std, confidence=.99)
    print("99% confidence: ", start, end)

    p_value = t_p_value(0, mean, n, n - 1, std)
    print("p value is: ", p_value)

def problem_set_9():
    mean = 1
    sigma = 5.5
    z_score = compute_z_score(0, mean, sigma)
    prob_greater_than_0 = cdf(z_score)
    print("probability >= 0: ", prob_greater_than_0)

    z_score_1 = compute_z_score(1, mean, sigma)
    print("probability between 0 and 1: ", cdf(z_score_1) - cdf(z_score))

def main():
    #problem_set_1()
    #problem_set_2()
    #problem_set_3()
    problem_set_4()
    #problem_set_5()
    #problem_set_6()
    #problem_set_7()
    #problem_set_8()
    #problem_set_9()

if __name__ == "__main__":
    main()