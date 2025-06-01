import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from common import standard_deviation, split_by, median, find_outliers, get_percentile_exclusive, get_iqr, histogram, compute_z_score, cdf
blue_df = pd.read_csv("./hw1/bluebonnet.csv")

df_dict = split_by(blue_df, "distance")
near = df_dict[0]
away = df_dict[1]
near_std = standard_deviation(near, "redpetals")
away_std = standard_deviation(away, "redpetals")
print("near variation: ", near_std)
print("away variation: ", away_std)
print(blue_df.describe())

print("median red petals near: ", median(near, "redpetals"))

outliers = find_outliers(near, "redpetals")
print(outliers)

print("average red petals near: ", near["redpetals"].mean())
print("average red petals away: ", away["redpetals"].mean())

print("petals < 4: ", get_percentile_exclusive(near, "redpetals", 4))
print("petals < 4 (away): ", get_percentile_exclusive(away, "redpetals", 4))

sat_df = pd.read_csv("./hw1/sat_gender.csv")
difference = sat_df["MaleVerbal"].median() - sat_df["FemaleVerbal"].median()
print("male - female: ", difference)
mean_difference = sat_df["MaleVerbal"].mean() - sat_df["FemaleVerbal"].mean()
print("male - female (mean): ", mean_difference)

print("male iqr: ", get_iqr(sat_df, "MaleVerbal"))
print("female iqr: ", get_iqr(sat_df, "FemaleVerbal"))

print("male std: ", standard_deviation(sat_df, "MaleVerbal"))
print("female std: ", standard_deviation(sat_df, "FemaleVerbal"))

print("male 90th: ", sat_df["MaleVerbal"].quantile(.9))
print("female 90th: ", sat_df["FemaleVerbal"].quantile(.9))

print("male outliers: ", find_outliers(sat_df, "MaleVerbal"))
print("female outliers: ", find_outliers(sat_df, "FemaleVerbal"))

surv_df = pd.read_csv("./hw1/heart_survival.csv")
#histogram(surv_df, "StandardTherapy")
print("standard outliers: ", find_outliers(surv_df, "StandardTherapy"))
print("new outliers: ", find_outliers(surv_df, "NewTherapy"))

print("standard less than 14 months: ", get_percentile_exclusive(surv_df, "StandardTherapy", 14))
print("new less than 14 months: ", get_percentile_exclusive(surv_df, "NewTherapy", 14))

stacked_df = pd.read_csv("./hw1/heart_survival_stack.csv")
print("total less than 14 months: ", get_percentile_exclusive(stacked_df, "Survival", 14))

rat_df = pd.read_csv("./hw1/adduct.csv")
corn_df = split_by(rat_df, "diet")["corn"]
fish_df = split_by(rat_df, "diet")["fish"]

print("corn average: ", corn_df["distal"].mean())
print("fish average: ", fish_df["distal"].mean())

print("corn median: ", corn_df["distal"].median())
print("fish median: ", fish_df["distal"].median())

print("corn std: ", corn_df["distal"].std())
print("fish std: ", fish_df["distal"].std())

print("corn iqr: ", get_iqr(corn_df, "distal"))
print("fish iqr: ", get_iqr(fish_df, "distal"))

print("corn outliers: ", find_outliers(corn_df, "distal"))
print("fish outliers: ", find_outliers(fish_df, "distal"))

z_score = compute_z_score(33, 17, 8)
cdf_33 = cdf(z_score)
print(f"probability greater than {z_score}: ", 1 - cdf(z_score))
z_score = compute_z_score(25, 17, 8)
cdf_25 = cdf(z_score)
print("probability between 25 and 33 is: ", cdf_33 - cdf_25)
print("probability less than 25: ", cdf_25)
print("original description: ", fish_df.describe())
rat_new_df = pd.read_csv("./hw1/adduct2.csv")
fish_new_df = split_by(rat_new_df, "diet")["fish"]
print("new description: ", fish_new_df.describe())