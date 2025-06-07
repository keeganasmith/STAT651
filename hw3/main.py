import pandas as pd
import sys
import os
from scipy.stats import ttest_ind

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from common import standard_deviation, split_by, median, find_outliers, get_percentile_exclusive, get_iqr, histogram, compute_z_score, cdf, confidence_interval_z, compute_required_sample_size, compute_standard_error, confidence_interval_t, t_test, t_p_value, compute_sample_size_power, compute_pooled_std
def p_set_1():
    df = pd.read_csv("./hw1/hormone_assay.csv")
    column = "Log(Test / Reference)"
    mean = df[column].mean()
    sample_size_50 = compute_sample_size_power(.01, .5, .05, .25)
    sample_size_80 = compute_sample_size_power(.01, .2, .05, .25)
    sample_size_99 = compute_sample_size_power(.01, .01, .05, .25)
    print("50% power: ", sample_size_50)
    print("80% power: ", sample_size_80)
    print("99% power: ", sample_size_99)

def p_set_2():
    df = pd.read_csv("./hw1/armspan.csv")
    df["difference"] = df["height"] - df["armspan"]
    mean = df["difference"].mean()
    print("mean: ", mean)
    std = df["difference"].std()
    lower, upper = confidence_interval_t(mean,len(df), std, .95)
    print("95% confidence for mean difference: ", lower, upper)

def p_set_3():
    df = pd.read_csv("./hw1/framingham.csv")
    column = "sbdiff"
    first_quantile = df[column].quantile(.25)
    print("first quantile is: ", first_quantile)
    third_quantile = df[column].quantile(.75)
    print("third quantile is: ", third_quantile)
    outliers = find_outliers(df, column, 3)
    print("outliers: ", outliers)

    lower, upper = confidence_interval_t(df[column].mean(), len(df), df[column].std(), .95)
    print("95 confidence: ", lower, upper)

def p_set_4():
    df = pd.read_csv("./hw1/fumigant.csv")
    f1_std = df["F1"].std()
    print("F1 std: ", f1_std)
    f2_std = df["F2"].std()
    print("F2 std: ", f2_std)
    pooled_std = compute_pooled_std(len(df), len(df), f1_std, f2_std)
    print("pooled: ", pooled_std)
    result = ttest_ind(df["F1"], df["F2"], equal_var=True)
    print(result)

def main():
    #p_set_1()
    #p_set_2()
    #p_set_3()
    p_set_4()
if __name__ == "__main__":
    main()