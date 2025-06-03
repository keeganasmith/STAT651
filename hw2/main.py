import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from common import standard_deviation, split_by, median, find_outliers, get_percentile_exclusive, get_iqr, histogram, compute_z_score, cdf, confidence_interval_z
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
    sample_mean = blue_df["redpetals"].mean()
    z_score = compute_z_score(sample_mean, 3, 1.5)
    print("z stat: ", z_score)
    lower, upper = confidence_interval_z(sample_mean, len(blue_df), 1.5, .95)
    print("confidence interval: ", lower, ",", upper)
    lower, upper = confidence_interval_z(sample_mean, len(blue_df), 1.5, .99)
    print("confidence interval: ", lower, ",", upper)

def main():
    #problem_set_1()
    #problem_set_2()
    problem_set_3()
if __name__ == "__main__":
    main()